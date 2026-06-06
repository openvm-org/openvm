use std::sync::Arc;

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{interaction::PermutationCheckBus, StarkProtocolConfig};

mod controller;
pub mod merkle;
pub mod offline_checker;
pub mod online;
pub mod persistent;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use controller::dimensions::ptr_bits_from_address_height;
pub use controller::*;
pub use online::{Address, AddressMap, INITIAL_TIMESTAMP};

use crate::{
    arch::{AirRefWithColumns, MemoryConfig},
    system::memory::{
        interface::MemoryInterfaceAirs, merkle::MemoryMerkleAir, offline_checker::MemoryBridge,
        persistent::PersistentBoundaryAir,
    },
};

// TODO: Currently this is only used for debug assertions, but we may switch to making it constant
// and removing from MemoryConfig.
// This is the max bit width of AS-native OpenVM memory pointers (i.e. pointers measured in
// AS-native storage cells, *not* in guest bytes).
//
// RV64 targets 2^32 byte-addressable memory. Since `RV64_MEMORY_AS` uses u16 storage cells, 2^32
// bytes equals 2^31 u16 cells, so the AS-native pointer width is 31 bits. See [`crate::arch`]'s
// `BYTE_POINTER_MAX_BITS` for the corresponding *byte*-pointer width (32).
pub const POINTER_MAX_BITS: usize = 31;

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

/// Number of little-endian 16-bit limbs used to represent an AS-native memory pointer on the
/// memory bus.
///
/// AS-native pointers can be up to [`POINTER_MAX_BITS`] (31) bits wide, which exceeds the
/// BabyBear field modulus. To avoid composing a full pointer into a single field element, every
/// memory-bus pointer is carried as two little-endian 16-bit limbs `[lo16, hi16]`.
pub const POINTER_LIMBS: usize = 2;

/// The full pointer to a location in memory consists of an address space and a pointer within
/// the address space.
///
/// The AS-native pointer is stored as little-endian 16-bit limbs `pointer_limbs = [lo16, hi16]`
/// (see [`POINTER_LIMBS`]). These limbs are *AS-native cell* pointer limbs, not guest byte-pointer
/// limbs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, AlignedBorrow, StructReflection)]
#[repr(C)]
pub struct MemoryAddress<S, T> {
    pub address_space: S,
    pub pointer_limbs: [T; POINTER_LIMBS],
}

impl<S, T> MemoryAddress<S, T> {
    pub fn new(address_space: S, pointer_limbs: [T; POINTER_LIMBS]) -> Self {
        Self {
            address_space,
            pointer_limbs,
        }
    }

    pub fn from<T1, T2>(a: MemoryAddress<T1, T2>) -> Self
    where
        T1: Into<S>,
        T2: Into<T>,
    {
        Self {
            address_space: a.address_space.into(),
            pointer_limbs: a.pointer_limbs.map(Into::into),
        }
    }
}

impl<S, T: openvm_stark_backend::p3_field::PrimeCharacteristicRing> MemoryAddress<S, T> {
    /// Build an address from a concrete AS-native `pointer`, split into little-endian 16-bit
    /// limbs `[lo16, hi16]`.
    #[inline(always)]
    pub fn from_u32_pointer(address_space: S, pointer: u32) -> Self {
        Self {
            address_space,
            pointer_limbs: pointer_to_limbs(pointer),
        }
    }
}

/// Splits a concrete AS-native pointer into little-endian 16-bit limbs `[lo16, hi16]`.
#[inline(always)]
pub fn pointer_to_limbs<T: openvm_stark_backend::p3_field::PrimeCharacteristicRing>(
    pointer: u32,
) -> [T; POINTER_LIMBS] {
    [T::from_u32(pointer & 0xffff), T::from_u32(pointer >> 16)]
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
        let memory_dims = mem_config.memory_dimensions();
        let boundary = PersistentBoundaryAir::<DIGEST_WIDTH> {
            memory_bus,
            merkle_bus,
            compression_bus,
            range_bus: bridge.range_bus(),
            memory_dimensions: memory_dims,
        };
        let merkle = MemoryMerkleAir::<DIGEST_WIDTH> {
            memory_dimensions: memory_dims,
            merkle_bus,
            compression_bus,
        };
        let interface = MemoryInterfaceAirs { boundary, merkle };
        Self { bridge, interface }
    }

    /// The order of memory AIRs is boundary, merkle (if exists)
    pub fn into_airs<SC: StarkProtocolConfig>(self) -> Vec<AirRefWithColumns<SC>> {
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
