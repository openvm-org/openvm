use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit_derive::AnyEnum;
use openvm_circuit_primitives::var_range::{VariableRangeCheckerAir, VariableRangeCheckerBus};
use openvm_instructions::SystemOpcode;
use openvm_poseidon2_air::Poseidon2SubAir;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    interaction::{LookupBus, PermutationCheckBus},
    AirRef,
};

pub mod connector;
pub mod memory;
// Necessary for the PublicValuesChip
pub mod native_adapter;
/// Chip to handle phantom instructions.
/// The Air will always constrain a NOP which advances pc by DEFAULT_PC_STEP.
/// The runtime executor will execute different phantom instructions that may
/// affect trace generation based on the operand.
pub mod phantom;
pub mod poseidon2;
pub mod program;
pub mod public_values;

use connector::VmConnectorAir;
use program::ProgramAir;
use public_values::PublicValuesAir;

use crate::{
    arch::{
        vm_poseidon2_config, AirInventory, AirInventoryError, BusIndexManager, ExecutionBridge,
        ExecutionBus, ExecutorInventory, ExecutorInventoryError, SystemConfig, VmAirWrapper,
        VmCircuitConfig, VmExecutionConfig,
    },
    system::{
        connector::VmConnectorChip,
        memory::{
            offline_checker::{MemoryBridge, MemoryBus},
            MemoryAirInventory, MemoryController,
        },
        native_adapter::NativeAdapterAir,
        phantom::PhantomAir,
        poseidon2::air::Poseidon2PeripheryAir,
        program::{ProgramBus, ProgramChip},
        public_values::{core::PublicValuesCoreAir, PublicValuesChip},
    },
};

/// **If** internal poseidon2 chip exists, then its periphery index is 0.
const POSEIDON2_EXT_AIR_IDX: usize = 1;

#[derive(AnyEnum, From)]
pub enum SystemExecutor<F> {
    PublicValues(PublicValuesChip<F>),
    // Phantom(PhantomChip<F>),
}

/// SystemPort combines system resources needed by most extensions
#[derive(Clone, Copy)]
pub struct SystemPort {
    pub execution_bus: ExecutionBus,
    pub program_bus: ProgramBus,
    pub memory_bridge: MemoryBridge,
}

#[derive(Clone)]
pub struct SystemAirInventory<SC: StarkGenericConfig> {
    pub config: SystemConfig,
    pub program: ProgramAir,
    pub connector: VmConnectorAir,
    pub memory: MemoryAirInventory<SC>,
    /// Public values AIR exists if and only if continuations is disabled and `num_public_values`
    /// is greater than 0.
    pub public_values: Option<PublicValuesAir>,
}

impl<SC: StarkGenericConfig> SystemAirInventory<SC> {
    pub fn new(
        config: SystemConfig,
        port: SystemPort,
        merkle_compression_buses: Option<(PermutationCheckBus, PermutationCheckBus)>,
    ) -> Self {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = port;
        let range_bus = memory_bridge.range_bus();
        let program = ProgramAir::new(program_bus);
        let connector = VmConnectorAir::new(
            execution_bus,
            program_bus,
            range_bus,
            config.memory_config.clk_max_bits,
        );
        assert_eq!(
            config.continuation_enabled,
            merkle_compression_buses.is_some()
        );

        let memory = MemoryAirInventory::new(
            memory_bridge,
            &config.memory_config,
            range_bus,
            merkle_compression_buses,
        );

        let public_values = if config.has_public_values_chip() {
            let air = VmAirWrapper::new(
                NativeAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                ),
                PublicValuesCoreAir::new(
                    config.num_public_values,
                    config.max_constraint_degree as u32 - 1,
                ),
            );
            Some(air)
        } else {
            None
        };

        Self {
            config,
            program,
            connector,
            memory,
            public_values,
        }
    }

    pub fn port(&self) -> SystemPort {
        SystemPort {
            memory_bridge: self.memory.bridge,
            program_bus: self.program.bus,
            execution_bus: self.connector.execution_bus,
        }
    }

    pub fn into_airs(self) -> Vec<AirRef<SC>> {
        let mut airs: Vec<AirRef<SC>> = Vec::new();
        airs.push(Arc::new(self.program));
        airs.push(Arc::new(self.connector));
        if let Some(public_values) = self.public_values {
            airs.push(Arc::new(public_values));
        }
        airs.extend(self.memory.into_airs());
        airs
    }
}

impl<F> VmExecutionConfig<F> for SystemConfig {
    type Executor = SystemExecutor<F>;

    fn create_executors(
        &self,
    ) -> Result<ExecutorInventory<Self::Executor, F>, ExecutorInventoryError> {
    }
}

impl<SC: StarkGenericConfig> VmCircuitConfig<SC> for SystemConfig {
    /// Every VM circuit within the OpenVM circuit architecture **must** be initialized from the
    /// [SystemConfig].
    fn create_circuit(&self) -> Result<AirInventory<SC>, AirInventoryError> {
        let mut bus_idx_mgr = BusIndexManager::new();
        let execution_bus = ExecutionBus::new(bus_idx_mgr.new_bus_idx());
        let memory_bus = MemoryBus::new(bus_idx_mgr.new_bus_idx());
        let program_bus = ProgramBus::new(bus_idx_mgr.new_bus_idx());
        let range_bus =
            VariableRangeCheckerBus::new(bus_idx_mgr.new_bus_idx(), self.memory_config.decomp);

        let merkle_compression_buses = if self.continuation_enabled {
            let merkle_bus = PermutationCheckBus::new(bus_idx_mgr.new_bus_idx());
            let compression_bus = PermutationCheckBus::new(bus_idx_mgr.new_bus_idx());
            Some((merkle_bus, compression_bus))
        } else {
            None
        };
        let memory_bridge =
            MemoryBridge::new(memory_bus, self.memory_config.clk_max_bits, range_bus);
        let system_port = SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        };
        let system = SystemAirInventory::new(self.clone(), system_port, merkle_compression_buses);

        let mut inventory = AirInventory::new(system, bus_idx_mgr);

        let range_checker = VariableRangeCheckerAir::new(range_bus);
        // Range checker is always the first AIR in the inventory
        inventory.add_air(range_checker);

        if self.continuation_enabled {
            assert_eq!(inventory.ext_airs().len(), POSEIDON2_EXT_AIR_IDX);
            // Add direct poseidon2 AIR for persistent memory.
            // Currently we never use poseidon2 opcodes when continuations is enabled: we will need
            // special handling when that happens
            let (_, compression_bus) = merkle_compression_buses.unwrap();
            let direct_bus_idx = compression_bus.index;
            let air = Poseidon2PeripheryAir::new(
                Arc::new(Poseidon2SubAir::new(vm_poseidon2_config().constants.into())),
                LookupBus::new(direct_bus_idx),
            );
            inventory.add_air(air);
        }
        let execution_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let phantom = PhantomAir {
            execution_bridge,
            phantom_opcode: SystemOpcode::PHANTOM.global_opcode(),
        };
        inventory.add_air(phantom);

        Ok(inventory)
    }
}

// =================== CPU Backend Specific System Chip Complex Constructor ==================

/// Base system chips for CPU backend. These chips must exactly correspond to the AIRs in
/// [SystemAirInventory]. The following don't execute instructions, but are essential
/// for the VM architecture.
pub struct SystemBase<F> {
    pub program_chip: ProgramChip<F>,
    pub connector_chip: VmConnectorChip<F>,
    /// Contains all memory chips
    pub memory_controller: MemoryController<F>,
    pub public_values: Option<PublicValuesChip<F>>,
}

impl<F: PrimeField32> SystemComplex<F> {
    pub fn new(config: SystemConfig) -> Self {
        let mut bus_idx_mgr = BusIndexManager::new();
        let execution_bus = ExecutionBus::new(bus_idx_mgr.new_bus_idx());
        let memory_bus = MemoryBus::new(bus_idx_mgr.new_bus_idx());
        let program_bus = ProgramBus::new(bus_idx_mgr.new_bus_idx());
        let range_bus =
            VariableRangeCheckerBus::new(bus_idx_mgr.new_bus_idx(), config.memory_config.decomp);

        let range_checker = SharedVariableRangeCheckerChip::new(range_bus);
        let memory_controller = if config.continuation_enabled {
            MemoryController::with_persistent_memory(
                memory_bus,
                config.memory_config.clone(),
                range_checker.clone(),
                PermutationCheckBus::new(bus_idx_mgr.new_bus_idx()),
                PermutationCheckBus::new(bus_idx_mgr.new_bus_idx()),
            )
        } else {
            MemoryController::with_volatile_memory(
                memory_bus,
                config.memory_config.clone(),
                range_checker.clone(),
            )
        };
        let memory_bridge = memory_controller.memory_bridge();
        let program_chip = ProgramChip::new(program_bus);
        let connector_chip = VmConnectorChip::new(
            execution_bus,
            program_bus,
            range_checker.clone(),
            config.memory_config.clk_max_bits,
        );

        let mut inventory = VmInventory::new();
        // PublicValuesChip is required when num_public_values > 0 in single segment mode.
        if config.has_public_values_chip() {
            assert_eq!(inventory.executors().len(), Self::PV_EXECUTOR_IDX);

            let chip = PublicValuesChip::new(
                VmAirWrapper::new(
                    NativeAdapterAir::new(
                        ExecutionBridge::new(execution_bus, program_bus),
                        memory_bridge,
                    ),
                    PublicValuesCoreAir::new(
                        config.num_public_values,
                        config.max_constraint_degree as u32 - 1,
                    ),
                ),
                PublicValuesCoreStep::new(
                    NativeAdapterStep::new(),
                    config.num_public_values,
                    config.max_constraint_degree as u32 - 1,
                ),
                memory_controller.helper(),
            );

            inventory
                .add_executor(chip, [PublishOpcode::PUBLISH.global_opcode()])
                .unwrap();
        }
        if config.continuation_enabled {
            assert_eq!(inventory.periphery().len(), Self::POSEIDON2_PERIPHERY_IDX);
            // Add direct poseidon2 chip for persistent memory.
            // This is **not** an instruction executor.
            // Currently we never use poseidon2 opcodes when continuations is enabled: we will need
            // special handling when that happens
            let direct_bus_idx = memory_controller
                .interface_chip
                .compression_bus()
                .unwrap()
                .index;
            let chip = Poseidon2PeripheryChip::new(
                vm_poseidon2_config(),
                direct_bus_idx,
                config.max_constraint_degree,
            );
            inventory.add_periphery_chip(chip);
        }
        let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
        let phantom_chip = PhantomChip::new(execution_bus, program_bus, SystemOpcode::CLASS_OFFSET);
        inventory
            .add_executor(RefCell::new(phantom_chip), [phantom_opcode])
            .unwrap();

        let base = SystemBase {
            program_chip,
            connector_chip,
            memory_controller,
            range_checker_chip: range_checker,
        };

        let max_trace_height = if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
            let min_log_blowup = log2_ceil_usize(config.max_constraint_degree - 1);
            1 << (BabyBear::TWO_ADICITY - min_log_blowup)
        } else {
            tracing::warn!(
                "constructing SystemComplex for unrecognized field; using max_trace_height = 2^30"
            );
            1 << 30
        };

        Self {
            config,
            base,
            inventory,
            bus_idx_mgr,
            overridden_inventory_heights: None,
            max_trace_height,
        }
    }
}
