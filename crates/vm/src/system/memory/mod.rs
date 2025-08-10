use std::sync::Arc;

use openvm_circuit_primitives::{is_less_than::IsLtSubAir, var_range::VariableRangeCheckerBus};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::PermutationCheckBus,
    p3_field::Field,
    p3_util::{log2_ceil_usize, log2_strict_usize},
    AirRef,
};

pub mod adapter;
mod controller;
pub mod merkle;
pub mod offline_checker;
pub mod online;
pub mod persistent;
#[cfg(test)]
mod tests;
pub mod volatile;

pub use controller::*;
pub use online::{Address, AddressMap, INITIAL_TIMESTAMP};

use crate::{
    arch::{MemoryConfig, ADDR_SPACE_OFFSET},
    system::memory::{
        adapter::AccessAdapterAir, dimensions::MemoryDimensions, interface::MemoryInterfaceAirs,
        merkle::MemoryMerkleAir, offline_checker::MemoryBridge, persistent::PersistentBoundaryAir,
        volatile::VolatileBoundaryAir,
    },
};

// @dev Currently this is only used for debug assertions, but we may switch to making it constant
// and removing from MemoryConfig
pub const POINTER_MAX_BITS: usize = 29;

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

/// The full pointer to a location in memory consists of an address space and a pointer within
/// the address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, AlignedBorrow)]
#[repr(C)]
pub struct MemoryAddress<S, T> {
    pub address_space: S,
    pub pointer: T,
}

impl<S, T> MemoryAddress<S, T> {
    pub fn new(address_space: S, pointer: T) -> Self {
        Self {
            address_space,
            pointer,
        }
    }

    pub fn from<T1, T2>(a: MemoryAddress<T1, T2>) -> Self
    where
        T1: Into<S>,
        T2: Into<T>,
    {
        Self {
            address_space: a.address_space.into(),
            pointer: a.pointer.into(),
        }
    }
}

#[derive(Clone)]
pub struct MemoryAirInventory<SC: StarkGenericConfig> {
    pub bridge: MemoryBridge,
    pub interface: MemoryInterfaceAirs,
    pub access_adapters: Vec<AirRef<SC>>,
}

impl<SC: StarkGenericConfig> MemoryAirInventory<SC> {
    pub fn new(
        bridge: MemoryBridge,
        mem_config: &MemoryConfig,
        range_bus: VariableRangeCheckerBus,
        merkle_compression_buses: Option<(PermutationCheckBus, PermutationCheckBus)>,
    ) -> Self {
        let memory_bus = bridge.memory_bus();
        let interface = if let Some((merkle_bus, compression_bus)) = merkle_compression_buses {
            // Persistent memory
            let memory_dims = MemoryDimensions {
                addr_space_height: mem_config.addr_space_height,
                address_height: mem_config.pointer_max_bits - log2_strict_usize(CHUNK),
            };
            let boundary = PersistentBoundaryAir::<CHUNK> {
                memory_dims,
                memory_bus,
                merkle_bus,
                compression_bus,
            };
            let merkle = MemoryMerkleAir::<CHUNK> {
                memory_dimensions: memory_dims,
                merkle_bus,
                compression_bus,
            };
            MemoryInterfaceAirs::Persistent { boundary, merkle }
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
            MemoryInterfaceAirs::Volatile { boundary }
        };
        // Memory access adapters
        let lt_air = IsLtSubAir::new(range_bus, mem_config.timestamp_max_bits);
        let maan = mem_config.max_access_adapter_n;
        assert!(matches!(maan, 2 | 4 | 8 | 16 | 32));
        let access_adapters: Vec<AirRef<SC>> = [
            Arc::new(AccessAdapterAir::<2> { memory_bus, lt_air }) as AirRef<SC>,
            Arc::new(AccessAdapterAir::<4> { memory_bus, lt_air }) as AirRef<SC>,
            Arc::new(AccessAdapterAir::<8> { memory_bus, lt_air }) as AirRef<SC>,
            Arc::new(AccessAdapterAir::<16> { memory_bus, lt_air }) as AirRef<SC>,
            Arc::new(AccessAdapterAir::<32> { memory_bus, lt_air }) as AirRef<SC>,
        ]
        .into_iter()
        .take(log2_strict_usize(maan))
        .collect();

        Self {
            bridge,
            interface,
            access_adapters,
        }
    }

    /// The order of memory AIRs is boundary, merkle (if exists), access adapters
    pub fn into_airs(self) -> Vec<AirRef<SC>> {
        let mut airs: Vec<AirRef<SC>> = Vec::new();
        match self.interface {
            MemoryInterfaceAirs::Volatile { boundary } => {
                airs.push(Arc::new(boundary));
            }
            MemoryInterfaceAirs::Persistent { boundary, merkle } => {
                airs.push(Arc::new(boundary));
                airs.push(Arc::new(merkle));
            }
        }
        airs.extend(self.access_adapters);
        airs
    }
}

/// This is O(1) and returns the length of
/// [`MemoryAirInventory::into_airs`].
pub fn num_memory_airs(is_persistent: bool, max_access_adapter_n: usize) -> usize {
    // boundary + { merkle if is_persistent } + access_adapters
    1 + usize::from(is_persistent) + log2_strict_usize(max_access_adapter_n)
}
