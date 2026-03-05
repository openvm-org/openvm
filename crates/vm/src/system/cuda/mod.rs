use std::sync::Arc;

use connector::VmConnectorChipGPU;
use memory::MemoryInventoryGPU;
use openvm_circuit::{
    arch::{DenseRecordArena, SystemConfig, PUBLIC_VALUES_AIR_ID},
    system::{
        connector::VmConnectorChip,
        memory::online::GuestMemory,
        SystemChipComplex, SystemRecords,
    },
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{prelude::F, GpuBackend};
use openvm_stark_backend::prover::{AirProvingContext, CommittedTraceData};
use poseidon2::Poseidon2PeripheryChipGPU;
use program::ProgramChipGPU;
use public_values::PublicValuesChipGPU;

use crate::system::memory::CHUNK;

pub(crate) const DIGEST_WIDTH: usize = 8;

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
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        hasher_chip: Arc<Poseidon2PeripheryChipGPU>,
    ) -> Self {
        let cpu_range_checker = range_checker.cpu_chip.clone().unwrap();

        // We create an empty program chip: the program should be loaded later (and can be swapped
        // out). The execution frequencies are supplied only after execution.
        let program_chip = ProgramChipGPU::new();
        let connector_chip = VmConnectorChipGPU::new(VmConnectorChip::new(
            cpu_range_checker.clone(),
            config.memory_config.timestamp_max_bits,
        ));

        let memory_inventory =
            MemoryInventoryGPU::persistent(config.memory_config.clone(), hasher_chip);

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
        self.memory_inventory.set_initial_memory(&memory.memory);
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

        let memory_ctxs = self.memory_inventory.generate_proving_ctxs(touched_memory);

        [program_ctx, connector_ctx]
            .into_iter()
            .chain(pv_ctx)
            .chain(memory_ctxs)
            .collect()
    }

    fn memory_top_tree(&self) -> Option<&[[F; CHUNK]]> {
        let top_tree = &self
            .memory_inventory
            .persistent
            .as_ref()
            .expect("GPU memory inventory must be persistent")
            .merkle_tree
            .top_roots_host;
        (!top_tree.is_empty()).then_some(top_tree.as_slice())
    }
}
