use std::sync::Arc;

use connector::VmConnectorChipGPU;
use memory::MemoryInventoryGPU;
use openvm_circuit::{
    arch::{DenseRecordArena, SystemConfig, PUBLIC_VALUES_AIR_ID},
    system::{
        connector::VmConnectorChip,
        memory::{interface::MemoryInterfaceAirs, online::GuestMemory, MemoryAirInventory},
        SystemChipComplex, SystemRecords,
    },
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{prover_backend::GpuBackend, types::F};
use openvm_stark_backend::{
    prover::types::{AirProvingContext, CommittedTraceData},
    Chip,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use poseidon2::Poseidon2PeripheryChipGPU;
use program::ProgramChipGPU;
use public_values::PublicValuesChipGPU;

pub(crate) const DIGEST_WIDTH: usize = 8;

pub mod access_adapters;
pub mod boundary;
pub mod connector;
pub mod extensions;
pub mod memory;
pub mod merkle_tree;
pub mod phantom;
pub mod poseidon2;
pub mod program;
pub mod public_values;

pub struct SystemChipInventoryGPU {
    pub program: ProgramChipGPU,
    pub connector: VmConnectorChipGPU,
    pub memory_inventory: MemoryInventoryGPU,
    pub public_values: Option<PublicValuesChipGPU>,
}

impl SystemChipInventoryGPU {
    pub fn new(
        config: &SystemConfig,
        mem_inventory: &MemoryAirInventory<BabyBearPoseidon2Config>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        hasher_chip: Option<Arc<Poseidon2PeripheryChipGPU>>,
    ) -> Self {
        let cpu_range_checker = range_checker.cpu_chip.clone().unwrap();

        // We create an empty program chip: the program should be loaded later (and can be swapped
        // out). The execution frequencies are supplied only after execution.
        let program_chip = ProgramChipGPU::new();
        let connector_chip = VmConnectorChipGPU::new(VmConnectorChip::new(
            cpu_range_checker.clone(),
            config.memory_config.timestamp_max_bits,
        ));

        let memory_inventory = match &mem_inventory.interface {
            MemoryInterfaceAirs::Persistent { .. } => {
                assert!(config.continuation_enabled);
                MemoryInventoryGPU::persistent(
                    config.memory_config.clone(),
                    range_checker.clone(),
                    hasher_chip.unwrap(),
                )
            }
            MemoryInterfaceAirs::Volatile { .. } => {
                assert!(!config.continuation_enabled);
                MemoryInventoryGPU::volatile(config.memory_config.clone(), range_checker.clone())
            }
        };

        let public_values_chip = config.has_public_values_chip().then(|| {
            PublicValuesChipGPU::new(
                range_checker,
                config.num_public_values,
                config.max_constraint_degree as u32 - 1,
                config.memory_config.timestamp_max_bits as u32,
            )
        });

        Self {
            program: program_chip,
            connector: connector_chip,
            memory_inventory,
            public_values: public_values_chip,
        }
    }
}

impl SystemChipComplex<DenseRecordArena, GpuBackend> for SystemChipInventoryGPU {
    fn load_program(&mut self, cached_program_trace: CommittedTraceData<GpuBackend>) {
        self.program.cached.replace(cached_program_trace);
    }

    fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        if self.memory_inventory.persistent.is_some() {
            self.memory_inventory.set_initial_memory(&memory.memory);
        }
    }

    fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<F>,
        mut record_arenas: Vec<DenseRecordArena>,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let SystemRecords {
            from_state,
            to_state,
            exit_code,
            filtered_exec_frequencies,
            access_adapter_records,
            touched_memory,
            public_values,
        } = system_records;

        let program_ctx = self.program.generate_proving_ctx(filtered_exec_frequencies);

        self.connector.cpu_chip.begin(from_state);
        self.connector.cpu_chip.end(to_state, exit_code);
        let connector_ctx = self.connector.generate_proving_ctx(());

        let pv_ctx = self.public_values.as_mut().map(|chip| {
            chip.public_values = public_values;
            let arena = record_arenas.remove(PUBLIC_VALUES_AIR_ID);
            chip.generate_proving_ctx(arena)
        });

        let memory_ctxs = self
            .memory_inventory
            .generate_proving_ctxs(access_adapter_records, touched_memory);

        [program_ctx, connector_ctx]
            .into_iter()
            .chain(pv_ctx)
            .chain(memory_ctxs)
            .collect()
    }

    #[cfg(feature = "metrics")]
    fn finalize_trace_heights(&self, heights: &mut [usize]) {
        use crate::system::cuda::boundary::BoundaryFields;

        let boundary_idx = PUBLIC_VALUES_AIR_ID + usize::from(self.public_values.is_some());
        let mut access_adapter_offset = boundary_idx + 1;
        match self.memory_inventory.boundary.fields {
            BoundaryFields::Volatile(_) => {
                let boundary_height = self.memory_inventory.boundary.num_records.unwrap_or(0);
                heights[boundary_idx] = boundary_height;
            }
            BoundaryFields::Persistent(ref boundary) => {
                let boundary_height = 2 * self.memory_inventory.boundary.num_records.unwrap_or(0);
                heights[boundary_idx] = boundary_height;
                heights[boundary_idx + 1] = self.memory_inventory.unpadded_merkle_height;
                access_adapter_offset += 1;

                // Poseidon2Periphery height also varies based on memory, so set it now even though
                // it's not a system chip:
                let poseidon_height = boundary
                    .poseidon2_buffer
                    .current_trace_height
                    .load(std::sync::atomic::Ordering::Relaxed);
                // We know the chip insertion index, which starts from *the end* of the the AIR
                // ordering
                const POSEIDON2_INSERTION_IDX: usize = 1;
                let poseidon_idx = heights.len() - 1 - POSEIDON2_INSERTION_IDX;
                heights[poseidon_idx] = poseidon_height;
            }
        }
        let access_heights = &self.memory_inventory.access_adapters.unpadded_heights;
        heights[access_adapter_offset..access_adapter_offset + access_heights.len()]
            .copy_from_slice(access_heights);
    }
}
