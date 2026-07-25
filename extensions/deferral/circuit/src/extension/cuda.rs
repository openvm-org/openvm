#[cfg(feature = "rvr")]
use std::any::Any;
use std::sync::Arc;

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
use openvm_cuda_backend::{
    prelude::F as CudaF, BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_riscv_circuit::Rv64ImGpuProverExt;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::{
        rvr::cuda::{
            GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
            RvrCheckpointAccessRegistry, RvrCheckpointAccessSpan,
        },
        rvr::RvrCheckpointPreflightExecution,
        GenerationError, MemoryConfig, VirtualMachine, MEMORY_BLOCK_BYTES,
    },
    openvm_cuda_common::stream::GpuDeviceCtx,
    openvm_deferral_transpiler::DeferralOpcode,
    openvm_instructions::{program::Program, riscv::RV64_MEMORY_AS, LocalOpcode},
    openvm_riscv_circuit::Rv64ImRvrGpuTracegen,
    openvm_stark_backend::p3_field::PrimeField32,
    openvm_stark_backend::prover::{AirProvingContext, ProvingContext},
};

use crate::{
    call::{DeferralCallAir, DeferralCallChipGpu},
    count::{DeferralCircuitCountAir, DeferralCircuitCountChipGpu},
    output::{DeferralOutputAir, DeferralOutputChipGpu},
    poseidon2::{DeferralPoseidon2Air, DeferralPoseidon2ChipGpu},
    DeferralExtension, Rv64DeferralConfig,
};

pub struct DeferralGpuProverExt;

/// Concrete arena-free Deferral + RV64/system checkpoint coordinator.
///
/// CALL expands its typed AS4 chronology. OUTPUT consumes its dynamic write
/// count and postimages directly from the checkpoint residual stream.
#[cfg(feature = "rvr")]
pub struct DeferralRvrGpuTracegen<'a> {
    program: &'a GpuRvrProgram,
    transcript: &'a GpuRvrTranscript,
    replay_plan: &'a GpuRvrReplayPlan,
    max_trace_height: usize,
    coverage: DeferralRvrCoverage,
}

#[cfg(feature = "rvr")]
struct DeferralRvrCoverage {
    pending_output: bool,
    pending_call: bool,
    pending_poseidon2: bool,
    pending_count: bool,
}

#[cfg(feature = "rvr")]
impl DeferralRvrCoverage {
    fn new() -> Self {
        Self {
            pending_output: true,
            pending_call: true,
            pending_poseidon2: true,
            pending_count: true,
        }
    }

    fn claim(pending: &mut bool, name: &str) -> Result<(), GpuRvrInputError> {
        if !std::mem::replace(pending, false) {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "Deferral RVR GPU tracegen visited duplicate {name} producer"
            )));
        }
        Ok(())
    }

    fn finish(self) -> Result<(), GpuRvrInputError> {
        let mut missing = Vec::new();
        if self.pending_output {
            missing.push("Output");
        }
        if self.pending_call {
            missing.push("Call");
        }
        if self.pending_poseidon2 {
            missing.push("Poseidon2");
        }
        if self.pending_count {
            missing.push("Count");
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(GpuRvrInputError::InvalidTranscript(format!(
                "Deferral RVR GPU tracegen did not visit producers {missing:?}"
            )))
        }
    }
}

#[cfg(feature = "rvr")]
impl<'a> DeferralRvrGpuTracegen<'a> {
    #[doc(hidden)]
    pub fn extension_opcodes() -> [u32; 2] {
        [
            DeferralOpcode::CALL.global_opcode().as_usize() as u32,
            DeferralOpcode::OUTPUT.global_opcode().as_usize() as u32,
        ]
    }

    #[doc(hidden)]
    pub fn register_checkpoint_access_schedules(
        registry: &mut RvrCheckpointAccessRegistry,
    ) -> Result<(), GpuRvrInputError> {
        registry.register(
            DeferralOpcode::CALL.global_opcode().as_usize() as u32,
            // CALL first reads the output and input heap pointers from rd/rs.
            &[1, 2],
            (1 << 6) | (1 << 7),
            4,
            5,
            &[
                RvrCheckpointAccessSpan::read_fixed(RV64_MEMORY_AS, 1, 4),
                RvrCheckpointAccessSpan::read_deferral_input_accumulator(3),
                RvrCheckpointAccessSpan::read_deferral_output_accumulator(3),
                RvrCheckpointAccessSpan::write_fixed_from_residuals(RV64_MEMORY_AS, 0, 5),
                RvrCheckpointAccessSpan::write_deferral_input_accumulator(3),
                RvrCheckpointAccessSpan::write_deferral_output_accumulator(3),
            ],
        )?;
        registry.register(
            DeferralOpcode::OUTPUT.global_opcode().as_usize() as u32,
            &[1, 2],
            (1 << 6) | (1 << 7),
            4,
            5,
            &[
                RvrCheckpointAccessSpan::read_fixed(RV64_MEMORY_AS, 1, 5),
                RvrCheckpointAccessSpan::write_count_from_residual_from_residuals(
                    RV64_MEMORY_AS,
                    0,
                    u32::MAX / MEMORY_BLOCK_BYTES as u32,
                ),
            ],
        )
    }

    pub fn upload_checkpoint_program<T: PrimeField32>(
        program: &Program<T>,
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
        vm.postflight(
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
        max_trace_height: usize,
    ) -> Result<Self, GpuRvrInputError> {
        Ok(Self {
            program,
            transcript,
            replay_plan,
            max_trace_height,
            coverage: DeferralRvrCoverage::new(),
        })
    }

    /// Handles exactly one Deferral-owned inventory chip. Returning `None`
    /// delegates the same reverse-walk position to RV64/system coverage.
    pub fn generate_for_chip(
        &mut self,
        chip: &dyn Any,
    ) -> Result<Option<AirProvingContext<GpuBackend>>, GpuRvrInputError> {
        if let Some(chip) = chip.downcast_ref::<DeferralOutputChipGpu>() {
            DeferralRvrCoverage::claim(&mut self.coverage.pending_output, "Output")?;
            return chip
                .generate_proving_ctx_from_rvr(
                    self.program,
                    self.transcript,
                    self.replay_plan,
                    self.max_trace_height,
                )
                .map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<DeferralCallChipGpu>() {
            DeferralRvrCoverage::claim(&mut self.coverage.pending_call, "Call")?;
            return chip
                .generate_proving_ctx_from_rvr(
                    self.program,
                    self.transcript,
                    self.replay_plan,
                    self.max_trace_height,
                )
                .map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<Arc<DeferralPoseidon2ChipGpu>>() {
            if self.coverage.pending_output || self.coverage.pending_call {
                return Err(GpuRvrInputError::InvalidTranscript(
                    "Deferral Poseidon2 producer was visited before executor producers".to_string(),
                ));
            }
            DeferralRvrCoverage::claim(&mut self.coverage.pending_poseidon2, "Poseidon2")?;
            return chip
                .generate_proving_ctx_direct(self.max_trace_height)
                .map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<Arc<DeferralCircuitCountChipGpu>>() {
            if self.coverage.pending_output
                || self.coverage.pending_call
                || self.coverage.pending_poseidon2
            {
                return Err(GpuRvrInputError::InvalidTranscript(
                    "Deferral Count producer was visited before dependent producers".to_string(),
                ));
            }
            DeferralRvrCoverage::claim(&mut self.coverage.pending_count, "Count")?;
            return chip
                .generate_proving_ctx_direct(self.max_trace_height)
                .map(Some);
        }
        Ok(None)
    }

    pub fn finish(self) -> Result<(), GpuRvrInputError> {
        self.coverage.finish()
    }

    /// Generates one complete Deferral + RV64/system segment in the VM's
    /// single reverse inventory walk, then verifies both coverage sets.
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
        let ctx = vm.generate_preflight_proving_ctx_unchecked_coverage(
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

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, DeferralExtension>
    for DeferralGpuProverExt
{
    fn extend_prover(
        &self,
        extension: &DeferralExtension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let num_deferral_circuits = extension.fns.len();
        let address_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        let count = Arc::new(if num_deferral_circuits == 0 {
            DeviceBuffer::<u32>::new()
        } else {
            DeviceBuffer::<u32>::with_capacity_on(num_deferral_circuits, &range_checker.device_ctx)
        });
        if num_deferral_circuits > 0 {
            count.fill_zero_on(&range_checker.device_ctx).unwrap();
        }

        inventory.next_air::<DeferralCircuitCountAir>()?;
        let count_chip = Arc::new(DeferralCircuitCountChipGpu::new(
            count.clone(),
            num_deferral_circuits,
            range_checker.device_ctx.clone(),
        ));
        inventory.add_periphery_chip(count_chip);

        inventory.next_air::<DeferralPoseidon2Air<CudaF>>()?;
        let poseidon2_chip = Arc::new(DeferralPoseidon2ChipGpu::new(
            1,
            range_checker.device_ctx.clone(),
        ));
        let poseidon2_shared = poseidon2_chip.shared_buffer();
        inventory.add_periphery_chip(poseidon2_chip);

        inventory.next_air::<DeferralCallAir>()?;
        let call_chip = DeferralCallChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            address_bits,
            timestamp_max_bits,
            count.clone(),
            num_deferral_circuits,
            poseidon2_shared.clone(),
        );
        inventory.add_executor_chip(call_chip);

        inventory.next_air::<DeferralOutputAir>()?;
        let output_chip = DeferralOutputChipGpu::new(
            range_checker,
            bitwise_lu,
            address_bits,
            timestamp_max_bits,
            count,
            num_deferral_circuits,
            poseidon2_shared,
        );
        inventory.add_executor_chip(output_chip);

        Ok(())
    }
}

#[derive(Clone)]
pub struct Rv64DeferralGpuBuilder;

impl VmBuilder<GpuBabyBearPoseidon2Engine> for Rv64DeferralGpuBuilder {
    type VmConfig = Rv64DeferralConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<GpuBabyBearPoseidon2Engine>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv64ImGpuProverExt,
            &config.rv64i,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv64ImGpuProverExt,
            &config.rv64m,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv64ImGpuProverExt,
            &config.io,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &DeferralGpuProverExt,
            &config.deferral,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests;
