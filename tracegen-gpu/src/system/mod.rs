use openvm_circuit::{
    arch::{DenseRecordArena, PUBLIC_VALUES_AIR_ID},
    system::{
        memory::{interface::MemoryInterface, online::GuestMemory, MemoryController},
        SystemChipComplex, SystemRecords,
    },
};
use openvm_stark_backend::{
    prover::types::{AirProvingContext, CommittedTraceData},
    Chip,
};
use stark_backend_gpu::{
    data_transporter::transport_matrix_to_device,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::system::{
    connector::VmConnectorChipGPU, program::ProgramChipGPU, public_values::PublicValuesChipGPU,
};

pub mod access_adapters;
pub mod boundary;
pub mod connector;
pub mod cuda;
pub mod phantom;
pub mod poseidon2;
pub mod program;
pub mod public_values;

pub struct SystemChipInventoryGPU {
    pub program: ProgramChipGPU,
    pub connector: VmConnectorChipGPU,
    pub memory: MemoryController<F>,
    pub public_values: Option<PublicValuesChipGPU>,
}

impl SystemChipComplex<DenseRecordArena, GpuBackend> for SystemChipInventoryGPU {
    fn load_program(&mut self, cached_program_trace: CommittedTraceData<GpuBackend>) {
        self.program.cached.replace(cached_program_trace);
    }

    fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        match &mut self.memory.interface_chip {
            MemoryInterface::Volatile { .. } => {}
            MemoryInterface::Persistent { initial_memory, .. } => {
                *initial_memory = memory.memory.clone();
            }
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

        self.program.filtered_exec_freqs = filtered_exec_frequencies;
        let program_ctx = self.program.generate_proving_ctx(());

        self.connector.begin(from_state);
        self.connector.end(to_state, exit_code);
        let connector_ctx = self.connector.generate_proving_ctx(());

        let pv_ctx = self.public_values.as_mut().map(|chip| {
            chip.public_values = public_values;
            let arena = record_arenas.remove(PUBLIC_VALUES_AIR_ID);
            chip.generate_proving_ctx(arena)
        });

        let cpu_memory_ctx = self
            .memory
            .generate_proving_ctx::<SC>(access_adapter_records, touched_memory);
        let memory_ctx = cpu_memory_ctx
            .into_iter()
            .map(|cpu_ctx| AirProvingContext::<GpuBackend> {
                cached_mains: vec![],
                common_main: Some(transport_matrix_to_device(cpu_ctx.common_main.unwrap())),
                public_values: cpu_ctx.public_values,
            })
            .collect::<Vec<_>>();

        [program_ctx, connector_ctx]
            .into_iter()
            .chain(pv_ctx)
            .chain(memory_ctx)
            .collect()
    }
}
