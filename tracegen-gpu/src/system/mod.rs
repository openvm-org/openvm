use std::sync::Arc;

use openvm_circuit::{
    arch::{DenseRecordArena, SystemConfig, PUBLIC_VALUES_AIR_ID},
    system::{
        connector::VmConnectorChip,
        memory::{
            interface::{MemoryInterface, MemoryInterfaceAirs},
            online::GuestMemory,
            MemoryAirInventory, MemoryController,
        },
        poseidon2::Poseidon2PeripheryChip,
        SystemChipComplex, SystemRecords,
    },
};
use openvm_stark_backend::{
    prover::types::{AirProvingContext, CommittedTraceData},
    Chip,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_baby_bear::BabyBear;
use stark_backend_gpu::{prover_backend::GpuBackend, types::F};

use crate::{
    cpu_proving_ctx_to_gpu,
    primitives::var_range::VariableRangeCheckerChipGPU,
    system::{
        connector::VmConnectorChipGPU, program::ProgramChipGPU, public_values::PublicValuesChipGPU,
    },
};

pub mod access_adapters;
pub mod boundary;
pub mod connector;
pub mod cuda;
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
    // TODO[arayi]: Switch to [MemoryInventoryGPU] once persistent memory is implemented
    pub memory_controller: MemoryController<BabyBear>,
    pub public_values: Option<PublicValuesChipGPU>,
}

impl SystemChipInventoryGPU {
    pub fn new(
        config: &SystemConfig,
        mem_inventory: &MemoryAirInventory<BabyBearPoseidon2Config>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        hasher_chip: Option<Arc<Poseidon2PeripheryChip<BabyBear>>>,
    ) -> Self {
        let cpu_range_checker = range_checker.cpu_chip.clone().unwrap();

        // We create an empty program chip: the program should be loaded later (and can be swapped
        // out). The execution frequencies are supplied only after execution.
        let program_chip = ProgramChipGPU::new();
        let connector_chip = VmConnectorChipGPU::new(VmConnectorChip::new(
            cpu_range_checker.clone(),
            config.memory_config.clk_max_bits,
        ));

        let memory_bus = mem_inventory.bridge.memory_bus();
        let memory_controller = match &mem_inventory.interface {
            MemoryInterfaceAirs::Persistent {
                boundary: _,
                merkle,
            } => {
                assert!(config.continuation_enabled);
                MemoryController::<BabyBear>::with_persistent_memory(
                    memory_bus,
                    config.memory_config.clone(),
                    cpu_range_checker,
                    merkle.merkle_bus,
                    merkle.compression_bus,
                    hasher_chip.unwrap(),
                )
            }
            MemoryInterfaceAirs::Volatile { boundary: _ } => {
                assert!(!config.continuation_enabled);
                MemoryController::with_volatile_memory(
                    memory_bus,
                    config.memory_config.clone(),
                    cpu_range_checker,
                )
            }
        };

        let public_values_chip = config.has_public_values_chip().then(|| {
            PublicValuesChipGPU::new(
                range_checker,
                config.num_public_values,
                config.max_constraint_degree as u32 - 1,
                config.memory_config.clk_max_bits as u32,
            )
        });

        Self {
            program: program_chip,
            connector: connector_chip,
            memory_controller,
            public_values: public_values_chip,
        }
    }
}

impl SystemChipComplex<DenseRecordArena, GpuBackend> for SystemChipInventoryGPU {
    fn load_program(&mut self, cached_program_trace: CommittedTraceData<GpuBackend>) {
        self.program.cached.replace(cached_program_trace);
    }

    fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        match &mut self.memory_controller.interface_chip {
            MemoryInterface::Volatile { .. } => {
                // Skip initialization for volatile memory
            }
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

        let program_ctx = self.program.generate_proving_ctx(filtered_exec_frequencies);

        self.connector.begin(from_state);
        self.connector.end(to_state, exit_code);
        let connector_ctx = self.connector.generate_proving_ctx(());

        let pv_ctx = self.public_values.as_mut().map(|chip| {
            chip.public_values = public_values;
            let arena = record_arenas.remove(PUBLIC_VALUES_AIR_ID);
            chip.generate_proving_ctx(arena)
        });

        let memory_ctx = self
            .memory_controller
            .generate_proving_ctx(access_adapter_records, touched_memory);

        let memory_ctxs = memory_ctx
            .into_iter()
            .map(cpu_proving_ctx_to_gpu)
            .collect::<Vec<_>>();

        [program_ctx, connector_ctx]
            .into_iter()
            .chain(pv_ctx)
            .chain(memory_ctxs)
            .collect()
    }
}
