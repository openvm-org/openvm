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
use openvm_instructions::{riscv::RV64_MEMORY_AS, VM_DIGEST_WIDTH};
use openvm_stark_backend::prover::{AirProvingContext, CommittedTraceData};
use poseidon2::Poseidon2PeripheryChipGPU;
use program::{DeviceProgramFrequenciesProvider, ProgramChipGPU};

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
    device_program_frequencies: Option<Arc<dyn DeviceProgramFrequenciesProvider>>,
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
            device_program_frequencies: None,
        }
    }

    pub fn set_device_touched_memory_provider(
        &mut self,
        provider: Arc<dyn DeviceTouchedMemoryProvider>,
    ) {
        self.device_touched_memory = Some(provider);
    }

    pub fn set_device_program_frequencies_provider(
        &mut self,
        provider: Arc<dyn DeviceProgramFrequenciesProvider>,
    ) {
        self.device_program_frequencies = Some(provider);
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
            program_frequencies_on_device,
            #[cfg(feature = "rvr")]
            rvr_exec_frequencies_touched,
            #[cfg(feature = "rvr")]
            rvr_exec_frequencies_pool,
            touched_memory,
            touched_memory_on_device,
            device_replay_oracle,
        } = system_records;

        let program_ctx = {
            let _span = tracing::info_span!("program_trace_gen").entered();
            if program_frequencies_on_device {
                let provider = self
                    .device_program_frequencies
                    .as_ref()
                    .expect("device program-frequency route has no provider");
                let initial_memory = self.memory_inventory.device_initial_memory();
                let frequencies = provider
                    .take_device_program_frequencies(
                        &self.memory_inventory.device_ctx,
                        &initial_memory,
                    )
                    .expect("device program-frequency route has no bound segment");
                self.program
                    .generate_proving_ctx_from_device_frequencies(frequencies)
            } else {
                self.program
                    .generate_proving_ctx_from_frequencies(&filtered_exec_frequencies)
            }
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
                .take_device_touched_memory(
                    &self.memory_inventory.device_ctx,
                    &self.memory_inventory.device_initial_memory(),
                )
                .expect("device touched-memory route has no bound segment");
            if device_replay_oracle {
                let actual = device_touched
                    .records
                    .to_host_on(&self.memory_inventory.device_ctx)
                    .expect("device touched-memory oracle D2H");
                let mut expected =
                    Vec::with_capacity(touched_memory.len() * DEVICE_TOUCHED_RECORD_WORDS);
                for touched in &touched_memory {
                    expected.push(touched.address_space);
                    expected.push(touched.ptr);
                    expected.push(touched.timestamp);
                    expected.extend(
                        touched
                            .values
                            .map(|value| unsafe { std::mem::transmute::<F, u32>(value) }),
                    );
                }
                assert_eq!(
                    &actual[..device_touched.num_records * DEVICE_TOUCHED_RECORD_WORDS],
                    expected.as_slice(),
                    "device touched-memory replay differs byte-for-byte from host replay"
                );
                eprintln!(
                    "OPENVM_RVR_DEVICE_REPLAY_ORACLE_PASS=1 touched_records={}",
                    device_touched.num_records
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

    fn merge_device_continuation_dirty_pages(&mut self, memory: &mut GuestMemory) {
        let Some(words) = self
            .device_touched_memory
            .as_ref()
            .and_then(|provider| provider.take_continuation_dirty_pages())
        else {
            return;
        };
        use openvm_circuit::system::memory::online::{LinearMemory, PAGE_SIZE};

        let address_space = RV64_MEMORY_AS as usize;
        let memory_bytes = memory
            .memory
            .mem
            .get(address_space)
            .expect("continuation dirty pages require main memory")
            .size();
        let page_count = memory_bytes.div_ceil(PAGE_SIZE);
        let expected_words = page_count.div_ceil(u64::BITS as usize);
        assert_eq!(
            words.len(),
            expected_words,
            "device continuation dirty-page bitmap geometry differs from carried memory"
        );
        if let Some(&last) = words.last() {
            let used = page_count % u64::BITS as usize;
            if used != 0 {
                assert_eq!(
                    last >> used,
                    0,
                    "device continuation dirty-page bitmap marks an out-of-range page"
                );
            }
        }
        let pages = &mut memory.memory.touched_pages[address_space];
        for (word_index, mut word) in words.into_iter().enumerate() {
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let page = word_index * u64::BITS as usize + bit;
                assert!(
                    page < page_count,
                    "device continuation page exceeds main memory"
                );
                pages.mark_byte_range(page * PAGE_SIZE, 1);
                word &= word - 1;
            }
        }
    }

    fn memory_top_tree(&self) -> Option<&[[F; VM_DIGEST_WIDTH]]> {
        let top_tree = &self.memory_inventory.merkle_tree.top_roots_host;
        (!top_tree.is_empty()).then_some(top_tree.as_slice())
    }
}
