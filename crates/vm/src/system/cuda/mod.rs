use std::sync::Arc;

use connector::VmConnectorChipGPU;
use memory::MemoryInventoryGPU;
use openvm_circuit::{
    arch::SystemConfig,
    system::{connector::VmConnectorChip, memory::online::GuestMemory, SystemChipComplex},
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{prelude::F, GpuBackend};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::prover::{AirProvingContext, CommittedTraceData};
use poseidon2::Poseidon2PeripheryChipGPU;
use program::ProgramChipGPU;

use crate::arch::cuda::postflight::{
    GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
};

pub mod boundary;
pub mod connector;
pub mod extensions;
pub mod memory;
pub mod merkle_tree;
pub mod phantom;
pub mod poseidon2;
pub mod program;

pub struct SystemChipInventoryGPU {
    pub program: ProgramChipGPU,
    pub connector: VmConnectorChipGPU,
    pub memory_inventory: MemoryInventoryGPU,
}

impl SystemChipInventoryGPU {
    pub fn new(
        config: &SystemConfig,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        hasher_chip: Arc<Poseidon2PeripheryChipGPU>,
        device_ctx: GpuDeviceCtx,
    ) -> Self {
        let cpu_range_checker = range_checker.cpu_chip.clone().unwrap();

        // We create an empty program chip: the program should be loaded later (and can be swapped
        // out). The execution frequencies are supplied only after execution.
        let program_chip = ProgramChipGPU::new(device_ctx.clone());
        let connector_chip = VmConnectorChipGPU::new(
            VmConnectorChip::new(
                cpu_range_checker.clone(),
                config.memory_config.timestamp_max_bits,
            ),
            device_ctx.clone(),
        );

        let memory_inventory = MemoryInventoryGPU::new(
            config.memory_config.clone(),
            range_checker,
            hasher_chip,
            device_ctx.clone(),
        );

        Self {
            program: program_chip,
            connector: connector_chip,
            memory_inventory,
        }
    }

    /// Generates every system AIR directly from one validated postflight segment.
    ///
    /// The initial memory image must already have been transported before
    /// preflight mutates the host state.
    pub fn generate_proving_ctx_from_postflight(
        &mut self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<Vec<AirProvingContext<GpuBackend>>, GpuPostflightError> {
        program.ensure_replay_inputs(transcript, replay_plan, &self.program.device_ctx)?;
        let program_ctx = {
            let _span = tracing::info_span!("program_trace_gen").entered();
            self.program.generate_proving_ctx_from_plan(replay_plan)?
        };

        let (from_state, to_state, exit_code) = replay_plan.connector_boundary();
        self.connector.cpu_chip.begin(from_state);
        self.connector.cpu_chip.end(to_state, exit_code);
        let connector_ctx = {
            let _span = tracing::info_span!("connector_trace_gen").entered();
            self.connector.generate_proving_ctx()
        };

        let memory_ctxs = self
            .memory_inventory
            .generate_proving_ctxs_from_transcript(transcript)?;
        Ok([program_ctx, connector_ctx]
            .into_iter()
            .chain(memory_ctxs)
            .collect())
    }
}

impl SystemChipComplex<GpuBackend> for SystemChipInventoryGPU {
    fn load_program(&mut self, cached_program_trace: CommittedTraceData<GpuBackend>) {
        self.program.cached.replace(cached_program_trace);
    }

    fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        self.memory_inventory.set_initial_memory(&memory.memory);
    }

    fn memory_top_tree(&self) -> Option<&[[F; VM_DIGEST_WIDTH]]> {
        let top_tree = &self.memory_inventory.merkle_tree.top_roots_host;
        (!top_tree.is_empty()).then_some(top_tree.as_slice())
    }
}
