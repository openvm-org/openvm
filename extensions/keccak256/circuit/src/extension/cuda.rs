use std::{
    any::Any,
    sync::{Arc, Mutex},
};

#[cfg(feature = "rvr")]
use openvm_circuit::arch::rvr::cuda::{
    PostflightAccessRegistry, PostflightAccessSchedule, PostflightAccessSpan,
};
use openvm_circuit::{
    arch::{
        cuda::postflight::{
            GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
        },
        prepare_gpu_postflight, to_byte_ptr_bits, GenerationError, PostflightTracegen,
        PreflightOutput, VirtualMachine, VmBuilder,
    },
    system::cuda::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemGpuBuilder,
        },
        SystemChipInventoryGPU,
    },
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_instructions::{program::Program, LocalOpcode};
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use openvm_riscv_circuit::{Rv64ImGpuProverExt, Rv64ImPreflightGpuTracegen};
use openvm_stark_backend::prover::{AirProvingContext, ProvingContext};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};
#[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
use {
    openvm_circuit::arch::{
        rvr::{cuda::PreflightReplayProgram, PreflightExecution},
        MemoryConfig,
    },
    openvm_cuda_common::stream::GpuDeviceCtx,
    openvm_stark_backend::p3_field::PrimeField32,
};

use super::*;
use crate::{
    cuda::{KeccakfOpChipGpu, KeccakfPermChipGpu, SharedKeccakfState, XorinVmChipGpu},
    keccakf_perm::KeccakfPermAir,
};

pub struct Keccak256GpuProverExt;

/// Keccak-owned immutable-history replay producers.
///
/// This is deliberately a concrete inventory helper rather than a generic tracegen
/// trait. The caller composes it with RV64 replay during the VM's existing reverse
/// inventory walk, then calls [`Self::finish`] to fail closed if either the op AIR or
/// its permutation AIR was skipped.
pub struct Keccak256PreflightGpuTracegen<'a> {
    program: &'a GpuPostflightProgram,
    transcript: &'a GpuPostflightTranscript,
    replay_plan: &'a GpuPostflightPlan,
    pending_xorin: bool,
    pending_keccakf_op: bool,
    pending_keccakf_perm: bool,
}

impl<'a> Keccak256PreflightGpuTracegen<'a> {
    #[doc(hidden)]
    pub fn extension_opcodes() -> [u32; 2] {
        [
            XorinOpcode::XORIN.global_opcode().as_usize() as u32,
            KeccakfOpcode::KECCAKF.global_opcode().as_usize() as u32,
        ]
    }

    #[doc(hidden)]
    #[cfg(feature = "rvr")]
    pub fn register_postflight_access_schedules(
        registry: &mut PostflightAccessRegistry,
    ) -> Result<(), GpuPostflightError> {
        registry.register(
            KeccakfOpcode::KECCAKF.global_opcode().as_usize() as u32,
            PostflightAccessSchedule {
                register_operands: &[1],
                zero_operand_mask: (1 << 2) | (1 << 3) | (1 << 6) | (1 << 7),
                register_as_operand: 4,
                memory_as_operand: 5,
                spans: &[PostflightAccessSpan::write_fixed_from_replay_values(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    0,
                    25,
                )],
            },
        )?;
        let count_shift = 3;
        let max_words = 17;
        registry.register(
            XorinOpcode::XORIN.global_opcode().as_usize() as u32,
            PostflightAccessSchedule {
                register_operands: &[1, 2, 3],
                zero_operand_mask: (1 << 6) | (1 << 7),
                register_as_operand: 4,
                memory_as_operand: 5,
                spans: &[
                    PostflightAccessSpan::read_count_from_register(
                        openvm_instructions::riscv::RV64_MEMORY_AS,
                        0,
                        2,
                        count_shift,
                        max_words,
                    ),
                    PostflightAccessSpan::read_count_from_register(
                        openvm_instructions::riscv::RV64_MEMORY_AS,
                        1,
                        2,
                        count_shift,
                        max_words,
                    ),
                    PostflightAccessSpan::write_register_count_from_replay_values(
                        openvm_instructions::riscv::RV64_MEMORY_AS,
                        0,
                        2,
                        count_shift,
                        max_words,
                    ),
                ],
            },
        )?;
        Ok(())
    }

    /// Uploads a program with the concrete RV64+Keccak checkpoint replay
    /// schedules installed once. Callers do not need to construct or merge the
    /// experimental registry themselves.
    #[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
    pub fn upload_postflight_program<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<PreflightReplayProgram, GpuPostflightError> {
        let mut registry = PostflightAccessRegistry::default();
        Self::register_postflight_access_schedules(&mut registry)?;
        registry
            .validate_no_native_collisions(Rv64ImPreflightGpuTracegen::postflight_opcode_bases())?;
        PreflightReplayProgram::upload_with_postflight_access_registry(
            program,
            memory_config,
            &registry,
            device_ctx,
        )
    }

    #[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
    pub fn postflight<VB>(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        program: &PreflightReplayProgram,
        execution: &PreflightExecution,
        num_insns: u32,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        let opcodes = Rv64ImPreflightGpuTracegen::postflight_opcode_bases();
        vm.postflight(program, execution, num_insns, opcodes)
    }

    pub fn new(
        program: &'a GpuPostflightProgram,
        transcript: &'a GpuPostflightTranscript,
        replay_plan: &'a GpuPostflightPlan,
    ) -> Self {
        let pending_xorin = !replay_plan
            .opcode_range(XorinOpcode::XORIN.global_opcode())
            .is_empty();
        let has_keccakf = !replay_plan
            .opcode_range(KeccakfOpcode::KECCAKF.global_opcode())
            .is_empty();
        Self {
            program,
            transcript,
            replay_plan,
            pending_xorin,
            pending_keccakf_op: has_keccakf,
            pending_keccakf_perm: has_keccakf,
        }
    }

    pub fn generate_for_chip(
        &mut self,
        chip: &dyn Any,
    ) -> Result<Option<AirProvingContext<GpuBackend>>, GpuPostflightError> {
        if let Some(chip) = chip.downcast_ref::<XorinVmChipGpu>() {
            self.pending_xorin = false;
            return chip
                .generate_proving_ctx_from_postflight(
                    self.program,
                    self.transcript,
                    self.replay_plan,
                )
                .map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<KeccakfOpChipGpu>() {
            self.pending_keccakf_op = false;
            return chip
                .generate_proving_ctx_from_postflight(
                    self.program,
                    self.transcript,
                    self.replay_plan,
                )
                .map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<KeccakfPermChipGpu>() {
            self.pending_keccakf_perm = false;
            return chip
                .generate_proving_ctx_from_postflight(
                    self.program,
                    self.transcript,
                    self.replay_plan,
                )
                .map(Some);
        }
        Ok(None)
    }

    pub fn finish(self) -> Result<(), GpuPostflightError> {
        let mut missing = Vec::new();
        if self.pending_xorin {
            missing.push("Xorin");
        }
        if self.pending_keccakf_op {
            missing.push("KeccakfOp");
        }
        if self.pending_keccakf_perm {
            missing.push("KeccakfPerm");
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(GpuPostflightError::InvalidTranscript(format!(
                "Keccak preflight GPU tracegen did not visit producers {missing:?}"
            )))
        }
    }

    /// Generates one complete RV64+Keccak segment through the VM's single
    /// reverse inventory walk and verifies both concrete producer sets.
    pub fn generate_proving_ctx<VB>(
        self,
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        let extension_opcodes = Self::extension_opcodes();
        let rv64 = Rv64ImPreflightGpuTracegen::new_after_claiming_extension_opcodes(
            self.program,
            self.transcript,
            self.replay_plan,
            &extension_opcodes,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        vm.generate_preflight_proving_ctx(
            self.program,
            self.transcript,
            self.replay_plan,
            (self, rv64),
            |(tracegen, rv64), chip| {
                if let Some(ctx) = tracegen
                    .generate_for_chip(chip)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?
                {
                    Ok(ctx)
                } else {
                    rv64.generate_for_chip(chip)
                        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
                }
            },
            |(tracegen, rv64)| {
                tracegen
                    .finish()
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
                rv64.finish()
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
            },
        )
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, Keccak256> for Keccak256GpuProverExt {
    fn extend_prover(
        &self,
        _extension: &Keccak256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // XorinVmChip
        inventory.next_air::<XorinVmAir>()?;
        let xorin_chip = XorinVmChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            byte_ptr_max_bits,
            timestamp_max_bits as u32,
        );
        inventory.add_executor_chip(xorin_chip);

        // Create shared state for passing records between Op and Perm chips
        let shared_state = Arc::new(Mutex::new(SharedKeccakfState::default()));

        // NOTE: AIRs are added in extend_circuit in this order: XorinVmAir, KeccakfPermAir,
        // KeccakfOpAir The prover extension must consume AIRs in the same order.

        // Register KeccakfPermChip (periphery chip - added BEFORE OpChip to ensure OpChip tracegen
        // runs first)
        inventory.next_air::<KeccakfPermAir>()?;
        let perm_chip =
            KeccakfPermChipGpu::new(shared_state.clone(), range_checker.device_ctx.clone());
        inventory.add_periphery_chip_with_height(perm_chip, None);

        // Register KeccakfOpChip (executor chip - generates first due to executor vs periphery
        // ordering)
        inventory.next_air::<KeccakfOpAir>()?;
        let op_chip = KeccakfOpChipGpu::new(
            range_checker,
            byte_ptr_max_bits,
            timestamp_max_bits as u32,
            shared_state,
        );
        inventory.add_executor_chip(op_chip);

        Ok(())
    }
}

#[derive(Clone)]
pub struct Keccak256Rv64GpuBuilder;

impl PostflightTracegen<GpuBabyBearPoseidon2Engine> for Keccak256Rv64GpuBuilder {
    type Prepared = GpuPostflightProgram;

    fn prepare_postflight(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, Self>,
        program: &Program<BabyBear>,
    ) -> Result<Self::Prepared, GenerationError> {
        prepare_gpu_postflight(vm, program)
    }

    fn generate_proving_ctx(
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, Self>,
        _host_program: &Program<BabyBear>,
        program: &Self::Prepared,
        output: &PreflightOutput,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let (transcript, replay_plan) = vm
            .postflight_history(program, output)
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        Keccak256PreflightGpuTracegen::new(program, &transcript, &replay_plan)
            .generate_proving_ctx(vm)
    }
}

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Keccak256Rv64GpuBuilder {
    type VmConfig = Keccak256Rv64Config;
    type SystemChipInventory = SystemChipInventoryGPU;

    fn create_chip_complex(
        &self,
        config: &Keccak256Rv64Config,
        circuit: AirInventory<<E as StarkEngine>::SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<<E as StarkEngine>::SC, <E as StarkEngine>::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _>::extend_prover(&Rv64ImGpuProverExt, &config.rv64i, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&Rv64ImGpuProverExt, &config.rv64m, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&Rv64ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _>::extend_prover(
            &Keccak256GpuProverExt,
            &config.keccak,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests;
