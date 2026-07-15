use std::sync::Arc;

use connector::VmConnectorChipGPU;
use memory::{DeviceTouchedMemoryProvider, MemoryInventoryGPU, DEVICE_TOUCHED_RECORD_WORDS};
use openvm_circuit::{
    arch::{DenseRecordArena, SystemConfig},
    system::{
        connector::VmConnectorChip, memory::online::GuestMemory, SystemChipComplex, SystemRecords,
    },
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyD2H, stream::GpuDeviceCtx};
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
    device_touched_memory: Option<Arc<dyn DeviceTouchedMemoryProvider>>,
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
            device_touched_memory: None,
        }
    }

    pub fn set_device_touched_memory_provider(
        &mut self,
        provider: Arc<dyn DeviceTouchedMemoryProvider>,
    ) {
        self.device_touched_memory = Some(provider);
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
            #[cfg(feature = "rvr")]
            rvr_exec_frequencies_touched,
            #[cfg(feature = "rvr")]
            rvr_exec_frequencies_pool,
            touched_memory,
            touched_memory_on_device,
        } = system_records;

        let program_ctx = {
            let _span = tracing::info_span!("program_trace_gen").entered();
            self.program
                .generate_proving_ctx_from_frequencies(&filtered_exec_frequencies)
        };
        #[cfg(feature = "rvr")]
        if let Some(pool) = rvr_exec_frequencies_pool {
            pool.recycle_exec_frequencies(filtered_exec_frequencies, rvr_exec_frequencies_touched);
        }

        self.connector.cpu_chip.begin(from_state);
        self.connector.cpu_chip.end(to_state, exit_code);
        let connector_ctx = {
            let _span = tracing::info_span!("connector_trace_gen").entered();
            self.connector.generate_proving_ctx(())
        };

        let memory_ctxs = if touched_memory_on_device {
            let provider = self
                .device_touched_memory
                .as_ref()
                .expect("device touched-memory route has no provider");
            let device_touched = provider
                .take_device_touched_memory(&self.memory_inventory.device_ctx)
                .expect("device touched-memory route has no bound segment");
            if std::env::var("OPENVM_RVR_DEVICE_REPLAY_ORACLE").as_deref() == Ok("1") {
                let actual = device_touched
                    .records
                    .to_host_on(&self.memory_inventory.device_ctx)
                    .expect("device touched-memory oracle D2H");
                let mut expected =
                    Vec::with_capacity(touched_memory.len() * DEVICE_TOUCHED_RECORD_WORDS);
                for &((address_space, ptr), timestamped) in &touched_memory {
                    expected.push(address_space);
                    expected.push(ptr);
                    expected.push(timestamped.timestamp);
                    expected.extend(
                        timestamped
                            .values
                            .map(|value| unsafe { std::mem::transmute::<F, u32>(value) }),
                    );
                }
                assert_eq!(
                    actual, expected,
                    "device touched-memory replay differs byte-for-byte from host replay"
                );
            }
            self.memory_inventory
                .generate_proving_ctxs_device(device_touched)
        } else {
            self.memory_inventory.generate_proving_ctxs(touched_memory)
        };

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
