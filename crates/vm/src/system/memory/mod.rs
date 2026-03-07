use std::sync::Arc;

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::PermutationCheckBus, p3_util::log2_strict_usize, AirRef, StarkProtocolConfig,
};

mod controller;
pub mod merkle;
pub mod offline_checker;
pub mod online;
pub mod persistent;
#[cfg(test)]
mod tests;

pub use controller::*;
pub use online::{Address, AddressMap, INITIAL_TIMESTAMP};

use crate::{
    arch::MemoryConfig,
    system::memory::{
        dimensions::MemoryDimensions, interface::MemoryInterfaceAirs, merkle::MemoryMerkleAir,
        offline_checker::MemoryBridge, persistent::PersistentBoundaryAir,
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
pub struct MemoryAirInventory {
    pub bridge: MemoryBridge,
    pub interface: MemoryInterfaceAirs,
}

impl MemoryAirInventory {
    pub fn new(
        bridge: MemoryBridge,
        mem_config: &MemoryConfig,
        merkle_bus: PermutationCheckBus,
        compression_bus: PermutationCheckBus,
    ) -> Self {
        let memory_bus = bridge.memory_bus();
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
        let interface = MemoryInterfaceAirs { boundary, merkle };
        Self { bridge, interface }
    }

    /// The order of memory AIRs is boundary, merkle (if exists)
    pub fn into_airs<SC: StarkProtocolConfig>(self) -> Vec<AirRef<SC>> {
        vec![
            Arc::new(self.interface.boundary),
            Arc::new(self.interface.merkle),
        ]
    }
}

/// This is O(1) and returns the length of
/// [`MemoryAirInventory::into_airs`].
pub const fn num_memory_airs() -> usize {
    // boundary + merkle
    2
}
