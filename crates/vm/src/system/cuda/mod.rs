use std::sync::Arc;

use connector::VmConnectorChipGPU;
use memory::MemoryInventoryGPU;
use openvm_circuit::{
    arch::{DenseRecordArena, SystemConfig},
    system::{
        connector::VmConnectorChip, memory::online::GuestMemory, SystemChipComplex, SystemRecords,
    },
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{prelude::F, GpuBackend};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::prover::{AirProvingContext, CommittedTraceData};
use poseidon2::Poseidon2PeripheryChipGPU;
use program::ProgramChipGPU;

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
            hasher_chip,
            device_ctx.clone(),
        );

        Self {
            program: program_chip,
            connector: connector_chip,
            memory_inventory,
        }
    }

    /// Generates every system AIR directly from one validated RVR segment.
    ///
    /// The initial memory image must already have been transported before RVR
    /// consumes and mutates the host state.
    #[cfg(feature = "rvr")]
    pub fn generate_proving_ctx_from_rvr(
        &mut self,
        program: &crate::arch::rvr::cuda::GpuRvrProgram,
        transcript: &crate::arch::rvr::cuda::GpuRvrTranscript,
        replay_plan: &crate::arch::rvr::cuda::GpuRvrReplayPlan,
    ) -> Result<Vec<AirProvingContext<GpuBackend>>, crate::arch::rvr::cuda::GpuRvrInputError> {
        program.ensure_replay_inputs(transcript, replay_plan, &self.program.device_ctx)?;
        let program_ctx = {
            let _span = tracing::info_span!("program_trace_gen").entered();
            // SAFETY: replay_plan owns this same-context buffer through the
            // entire system tracegen call. Memory tracegen below synchronizes
            // the same stream before returning.
            unsafe {
                self.program
                    .generate_proving_ctx_from_device(replay_plan.program_frequencies())
            }
        };

        let (from_state, to_state, exit_code) = replay_plan.connector_boundary();
        self.connector.cpu_chip.begin(from_state);
        self.connector.cpu_chip.end(to_state, exit_code);
        let connector_ctx = {
            let _span = tracing::info_span!("connector_trace_gen").entered();
            self.connector.generate_proving_ctx(())
        };

        // SAFETY: transcript owns the validated initialized prefix and remains
        // borrowed until this synchronous memory-inventory call returns.
        let memory_ctxs = unsafe {
            self.memory_inventory.generate_proving_ctxs_from_device(
                transcript.touched_blocks(),
                transcript.num_touched_blocks(),
            )
        };
        Ok([program_ctx, connector_ctx]
            .into_iter()
            .chain(memory_ctxs)
            .collect())
    }
}

impl SystemChipComplex<DenseRecordArena, GpuBackend> for SystemChipInventoryGPU {
    fn load_program(&mut self, cached_program_trace: CommittedTraceData<GpuBackend>) {
        self.program.cached.replace(cached_program_trace);
    }

    fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        self.memory_inventory.set_initial_memory(&memory.memory);
    }

    fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<F>,
        _record_arenas: Vec<DenseRecordArena>,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let SystemRecords {
            from_state,
            to_state,
            exit_code,
            filtered_exec_frequencies,
            touched_memory,
        } = system_records;

        let program_ctx = {
            let _span = tracing::info_span!("program_trace_gen").entered();
            self.program.generate_proving_ctx(filtered_exec_frequencies)
        };

        self.connector.cpu_chip.begin(from_state);
        self.connector.cpu_chip.end(to_state, exit_code);
        let connector_ctx = {
            let _span = tracing::info_span!("connector_trace_gen").entered();
            self.connector.generate_proving_ctx(())
        };

        let memory_ctxs = self.memory_inventory.generate_proving_ctxs(touched_memory);

        [program_ctx, connector_ctx]
            .into_iter()
            .chain(memory_ctxs)
            .collect()
    }

    fn memory_top_tree(&self) -> Option<&[[F; VM_DIGEST_WIDTH]]> {
        let top_tree = &self.memory_inventory.merkle_tree.top_roots_host;
        (!top_tree.is_empty()).then_some(top_tree.as_slice())
    }
}
