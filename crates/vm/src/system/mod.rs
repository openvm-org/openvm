use std::sync::Arc;

use openvm_circuit_primitives::{is_less_than::IsLtSubAir, var_range::VariableRangeCheckerBus};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::PermutationCheckBus,
    p3_field::Field,
    p3_util::{log2_ceil_usize, log2_strict_usize},
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
    arch::{ExecutionBridge, ExecutionBus, SystemConfig, VmAirWrapper, ADDR_SPACE_OFFSET},
    system::{
        memory::{
            adapter::AccessAdapterAir, dimensions::MemoryDimensions, merkle::MemoryMerkleAir,
            offline_checker::MemoryBridge, persistent::PersistentBoundaryAir,
            volatile::VolatileBoundaryAir, CHUNK,
        },
        native_adapter::NativeAdapterAir,
        program::ProgramBus,
        public_values::core::PublicValuesCoreAir,
    },
};

/// SystemPort combines system resources needed by most extensions
#[derive(Clone, Copy)]
pub struct SystemPort {
    pub execution_bus: ExecutionBus,
    pub program_bus: ProgramBus,
    pub memory_bridge: MemoryBridge,
}

#[derive(Clone)]
pub struct SystemAirs<SC: StarkGenericConfig> {
    pub config: SystemConfig,
    pub program: ProgramAir,
    pub connector: VmConnectorAir,
    pub memory_bridge: MemoryBridge,
    /// The order of memory AIRs is boundary, merkle (if exists), access adapters
    pub memory: Vec<AirRef<SC>>,
    /// Public values AIR exists if and only if continuations is disabled and `num_public_values`
    /// is greater than 0.
    pub public_values: Option<PublicValuesAir>,
}

impl<SC: StarkGenericConfig> SystemAirs<SC> {
    pub fn new(
        config: SystemConfig,
        port: SystemPort,
        range_bus: VariableRangeCheckerBus,
        merkle_compression_bus: Option<(PermutationCheckBus, PermutationCheckBus)>,
    ) -> Self {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = port;
        let program = ProgramAir::new(program_bus);
        let connector = VmConnectorAir::new(
            execution_bus,
            program_bus,
            range_bus,
            config.memory_config.clk_max_bits,
        );
        assert_eq!(
            config.continuation_enabled,
            merkle_compression_bus.is_some()
        );
        let memory_bus = memory_bridge.memory_bus();
        let mem_config = &config.memory_config;
        // TODO: consider making MemoryAirInventory to hold this
        let mut memory: Vec<AirRef<SC>> = Vec::new();
        if let Some((merkle_bus, compression_bus)) = merkle_compression_bus {
            // Persistent memory
            let memory_dims = MemoryDimensions {
                addr_space_height: mem_config.addr_space_height,
                address_height: mem_config.pointer_max_bits - log2_strict_usize(CHUNK),
            };
            let boundary = PersistentBoundaryAir::<CHUNK>::new(
                memory_dims,
                memory_bus,
                merkle_bus,
                compression_bus,
            );
            let merkle = MemoryMerkleAir::<CHUNK>::new(memory_dims, merkle_bus, compression_bus);
            memory.push(Arc::new(boundary));
            memory.push(Arc::new(merkle));
        } else {
            // Volatile memory
            let addr_space_height = mem_config.addr_space_height;
            assert!(addr_space_height < Val::<SC>::bits() - 2);
            let addr_space_max_bits =
                log2_ceil_usize((ADDR_SPACE_OFFSET + 2u32.pow(addr_space_height as u32)) as usize);
            let boundary = VolatileBoundaryAir::new(
                memory_bus,
                addr_space_max_bits,
                mem_config.pointer_max_bits,
                range_bus,
            );
            memory.push(Arc::new(boundary));
        }
        // Memory access adapters
        let lt_air = IsLtSubAir::new(range_bus, config.memory_config.clk_max_bits);
        let maan = mem_config.max_access_adapter_n;
        assert!(matches!(maan, 2 | 4 | 8 | 16 | 32));
        memory.extend(
            [
                Arc::new(AccessAdapterAir::<2> { memory_bus, lt_air }) as AirRef<SC>,
                Arc::new(AccessAdapterAir::<4> { memory_bus, lt_air }) as AirRef<SC>,
                Arc::new(AccessAdapterAir::<8> { memory_bus, lt_air }) as AirRef<SC>,
                Arc::new(AccessAdapterAir::<16> { memory_bus, lt_air }) as AirRef<SC>,
                Arc::new(AccessAdapterAir::<32> { memory_bus, lt_air }) as AirRef<SC>,
            ]
            .into_iter()
            .take(log2_strict_usize(maan)),
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
            memory_bridge,
            program,
            connector,
            memory,
            public_values,
        }
    }

    pub fn port(&self) -> SystemPort {
        SystemPort {
            memory_bridge: self.memory_bridge,
            program_bus: self.program.bus,
            execution_bus: self.connector.execution_bus,
        }
    }
}
