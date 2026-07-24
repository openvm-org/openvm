use openvm_circuit::{
    arch::{
        to_byte_ptr_bits, AirInventory, ChipInventory, ChipInventoryError, DenseRecordArena,
        VmBuilder, VmChipComplex, VmProverExtension,
    },
    system::cuda::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemGpuBuilder,
        },
        SystemChipInventoryGPU,
    },
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_riscv_circuit::Rv64ImGpuProverExt;
use openvm_sha2_air::{Sha256Config, Sha512Config};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::{
        rvr::{
            cuda::{
                GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
                RvrCheckpointAccessRegistry, RvrCheckpointAccessSpan,
            },
            RvrCheckpointPreflightExecution,
        },
        GenerationError, MemoryConfig, VirtualMachine,
    },
    openvm_circuit_primitives::AnyChip,
    openvm_cuda_common::stream::GpuDeviceCtx,
    openvm_instructions::{program::Program, riscv::RV64_MEMORY_AS, LocalOpcode},
    openvm_riscv_circuit::Rv64ImRvrGpuTracegen,
    openvm_sha2_transpiler::Rv64Sha2Opcode,
    openvm_stark_backend::{
        p3_field::PrimeField32,
        prover::{AirProvingContext, ProvingContext},
    },
};

use super::*;
use crate::{
    cuda::{Sha2BlockHasherChipGpu, Sha2MainChipGpu},
    Sha2BlockHasherVmAir, Sha2MainAir,
};

pub struct Sha2GpuProverExt;

/// Concrete checkpoint-replay composition for RV64 plus the two SHA-2 opcodes.
///
/// A SHA instruction has two trace producers: the main AIR owns execution and
/// memory interactions, while the block-hasher AIR owns compression. Both must
/// be visited in the VM's existing reverse inventory walk.
#[cfg(feature = "rvr")]
pub struct Sha2RvrGpuTracegen<'a> {
    program: &'a GpuRvrProgram,
    transcript: &'a GpuRvrTranscript,
    replay_plan: &'a GpuRvrReplayPlan,
    pending_sha256_main: bool,
    pending_sha256_block: bool,
    pending_sha512_main: bool,
    pending_sha512_block: bool,
}

#[cfg(feature = "rvr")]
impl<'a> Sha2RvrGpuTracegen<'a> {
    #[doc(hidden)]
    pub fn extension_opcodes() -> [u32; 2] {
        [
            Rv64Sha2Opcode::SHA256.global_opcode().as_usize() as u32,
            Rv64Sha2Opcode::SHA512.global_opcode().as_usize() as u32,
        ]
    }

    #[doc(hidden)]
    pub fn register_checkpoint_access_schedules(
        registry: &mut RvrCheckpointAccessRegistry,
    ) -> Result<(), GpuRvrInputError> {
        for (opcode, input_blocks, state_blocks) in [
            (Rv64Sha2Opcode::SHA256, 8, 4),
            (Rv64Sha2Opcode::SHA512, 16, 8),
        ] {
            registry.register(
                opcode.global_opcode().as_usize() as u32,
                &[1, 2, 3],
                (1 << 6) | (1 << 7),
                4,
                5,
                &[
                    RvrCheckpointAccessSpan::read_fixed(RV64_MEMORY_AS, 2, input_blocks),
                    RvrCheckpointAccessSpan::read_fixed(RV64_MEMORY_AS, 1, state_blocks),
                    RvrCheckpointAccessSpan::write_fixed_from_residuals(
                        RV64_MEMORY_AS,
                        0,
                        state_blocks,
                    ),
                ],
            )?;
        }
        Ok(())
    }

    pub fn upload_checkpoint_program<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<GpuRvrProgram, GpuRvrInputError> {
        let mut registry = RvrCheckpointAccessRegistry::default();
        Self::register_checkpoint_access_schedules(&mut registry)?;
        registry.validate_no_native_collisions(Rv64ImRvrGpuTracegen::checkpoint_opcode_bases())?;
        GpuRvrProgram::upload_with_checkpoint_access_registry(
            program,
            memory_config,
            &registry,
            device_ctx,
        )
    }

    pub fn expand_checkpoint_replay<VB>(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        program: &GpuRvrProgram,
        execution: &RvrCheckpointPreflightExecution,
        expected_retired: u32,
    ) -> Result<(GpuRvrTranscript, GpuRvrReplayPlan), GpuRvrInputError>
    where
        VB: VmBuilder<
            GpuBabyBearPoseidon2Engine,
            RecordArena = DenseRecordArena,
            SystemChipInventory = SystemChipInventoryGPU,
        >,
    {
        vm.expand_rvr_checkpoint_replay(
            program,
            execution,
            expected_retired,
            Rv64ImRvrGpuTracegen::checkpoint_opcode_bases(),
        )
    }

    pub fn new(
        program: &'a GpuRvrProgram,
        transcript: &'a GpuRvrTranscript,
        replay_plan: &'a GpuRvrReplayPlan,
    ) -> Self {
        let has_sha256 = !replay_plan
            .opcode_range(Rv64Sha2Opcode::SHA256.global_opcode())
            .is_empty();
        let has_sha512 = !replay_plan
            .opcode_range(Rv64Sha2Opcode::SHA512.global_opcode())
            .is_empty();
        Self {
            program,
            transcript,
            replay_plan,
            pending_sha256_main: has_sha256,
            pending_sha256_block: has_sha256,
            pending_sha512_main: has_sha512,
            pending_sha512_block: has_sha512,
        }
    }

    /// Returns `Some` only for a SHA-owned producer, allowing RV64 to handle
    /// every other chip in the same reverse inventory walk.
    pub fn generate_for_chip(
        &mut self,
        chip: &dyn AnyChip<DenseRecordArena, GpuBackend>,
    ) -> Result<Option<AirProvingContext<GpuBackend>>, GpuRvrInputError> {
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<Sha2MainChipGpu<Sha256Config>>()
        {
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.pending_sha256_main = false;
            return Ok(Some(ctx));
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<Sha2BlockHasherChipGpu<Sha256Config>>()
        {
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.pending_sha256_block = false;
            return Ok(Some(ctx));
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<Sha2MainChipGpu<Sha512Config>>()
        {
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.pending_sha512_main = false;
            return Ok(Some(ctx));
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<Sha2BlockHasherChipGpu<Sha512Config>>()
        {
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.pending_sha512_block = false;
            return Ok(Some(ctx));
        }
        Ok(None)
    }

    pub fn finish(self) -> Result<(), GpuRvrInputError> {
        let mut missing = Vec::new();
        if self.pending_sha256_main {
            missing.push("Sha256Main");
        }
        if self.pending_sha256_block {
            missing.push("Sha256BlockHasher");
        }
        if self.pending_sha512_main {
            missing.push("Sha512Main");
        }
        if self.pending_sha512_block {
            missing.push("Sha512BlockHasher");
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(GpuRvrInputError::InvalidTranscript(format!(
                "SHA-2 RVR GPU tracegen did not visit producers {missing:?}"
            )))
        }
    }

    /// Generates one complete RV64+SHA-2 segment and verifies that every
    /// executed opcode reached all of its concrete trace producers.
    pub fn generate_proving_ctx<VB>(
        mut self,
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError>
    where
        VB: VmBuilder<
            GpuBabyBearPoseidon2Engine,
            RecordArena = DenseRecordArena,
            SystemChipInventory = SystemChipInventoryGPU,
        >,
    {
        let extension_opcodes = Self::extension_opcodes();
        let mut rv64 = Rv64ImRvrGpuTracegen::new_after_claiming_extension_opcodes(
            self.program,
            self.transcript,
            self.replay_plan,
            &extension_opcodes,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let ctx = vm.generate_proving_ctx_from_rvr_unchecked_coverage(
            self.program,
            self.transcript,
            self.replay_plan,
            |insertion_idx, chip| {
                if let Some(ctx) = self
                    .generate_for_chip(chip)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?
                {
                    Ok(ctx)
                } else {
                    rv64.generate_for_chip(insertion_idx, chip)
                        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
                }
            },
        )?;
        self.finish()
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        rv64.finish()
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        vm.complete_rvr_tracegen_session();
        Ok(ctx)
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Sha2> for Sha2GpuProverExt {
    fn extend_prover(
        &self,
        _: &Sha2,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker_gpu = get_inventory_range_checker(inventory);
        let bitwise_gpu = get_or_create_bitwise_op_lookup(inventory)?;

        // SHA-256
        inventory.next_air::<Sha2BlockHasherVmAir<Sha256Config>>()?;
        let sha256_shared_records = Arc::new(Mutex::new(None));
        let sha256_block_gpu = Sha2BlockHasherChipGpu::<Sha256Config>::new(
            sha256_shared_records.clone(),
            bitwise_gpu.clone(),
            range_checker_gpu.clone(),
            byte_ptr_max_bits as u32,
        );
        inventory.add_periphery_chip(sha256_block_gpu);

        inventory.next_air::<Sha2MainAir<Sha256Config>>()?;
        let sha256_main_gpu = Sha2MainChipGpu::<Sha256Config>::new(
            sha256_shared_records,
            range_checker_gpu.clone(),
            byte_ptr_max_bits as u32,
            timestamp_max_bits as u32,
        );
        inventory.add_executor_chip(sha256_main_gpu);

        // SHA-512 (also covers SHA-384 constraints)
        inventory.next_air::<Sha2BlockHasherVmAir<Sha512Config>>()?;
        let sha512_shared_records = Arc::new(Mutex::new(None));
        let sha512_block_gpu = Sha2BlockHasherChipGpu::<Sha512Config>::new(
            sha512_shared_records.clone(),
            bitwise_gpu.clone(),
            range_checker_gpu.clone(),
            byte_ptr_max_bits as u32,
        );
        inventory.add_periphery_chip(sha512_block_gpu);

        inventory.next_air::<Sha2MainAir<Sha512Config>>()?;
        let sha512_main_gpu = Sha2MainChipGpu::<Sha512Config>::new(
            sha512_shared_records,
            range_checker_gpu,
            byte_ptr_max_bits as u32,
            timestamp_max_bits as u32,
        );
        inventory.add_executor_chip(sha512_main_gpu);

        Ok(())
    }
}

pub struct Sha2Rv64GpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Sha2Rv64GpuBuilder {
    type VmConfig = Sha2Rv64Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Sha2Rv64Config,
        circuit: AirInventory<<E as StarkEngine>::SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<
            <E as StarkEngine>::SC,
            Self::RecordArena,
            <E as StarkEngine>::PB,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.rv64i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.rv64m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Sha2GpuProverExt, &config.sha2, inventory)?;
        Ok(chip_complex)
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests;
