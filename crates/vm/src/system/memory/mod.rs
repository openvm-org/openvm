use std::sync::Arc;

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{BLOCK_FE_WIDTH, VM_DIGEST_WIDTH};
use openvm_platform::memory::MEM_BITS;
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
    arch::{AirRefWithColumns, MemoryConfig, U16_CELL_SIZE_BITS},
    system::memory::{
        interface::MemoryInterfaceAirs, merkle::MemoryMerkleAir, offline_checker::MemoryBridge,
        persistent::PersistentBoundaryAir,
    },
};

/// Default maximum bit width of pointers within each address space. Pointers index cells, not
/// bytes.
pub const DEFAULT_POINTER_MAX_BITS: usize = MEM_BITS - U16_CELL_SIZE_BITS;
// Valid RVR memory pointers and leaf indices fit in `u32`. Guest operands stay
// `u64` until a runtime bounds check proves that they are valid pointers.
const _: () = assert!(MEM_BITS <= u32::BITS as usize);

/// Returns whether `bytes` contains any non-zero byte.
///
/// Fixed-size comparisons allow the compiler to vectorize scans of mostly-zero memory pages.
#[inline]
pub(crate) fn has_nonzero_byte(bytes: &[u8]) -> bool {
    const ZERO_CHUNK: [u8; 32] = [0; 32];

    let mut chunks = bytes.chunks_exact(ZERO_CHUNK.len());
    chunks.any(|chunk| chunk != ZERO_CHUNK) || chunks.remainder().iter().any(|&byte| byte != 0)
}

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

/// Number of low pointer bits omitted from a memory-bus address.
///
/// Every memory-bus operation addresses one [`BLOCK_FE_WIDTH`]-cell block, and block starts are
/// aligned to [`BLOCK_FE_WIDTH`]. The bus therefore carries the block index `pointer /
/// BLOCK_FE_WIDTH` instead of the AS-native cell pointer.
pub const MEMORY_BLOCK_INDEX_SHIFT: usize = BLOCK_FE_WIDTH.ilog2() as usize;

/// The full pointer to a location in memory consists of an address space and a pointer within
/// the address space.
///
/// The memory bus addresses [`BLOCK_FE_WIDTH`]-cell blocks, so the pointer is expressed at block
/// granularity: an AS-native cell pointer divided by [`BLOCK_FE_WIDTH`]. With the largest
/// supported 32-bit AS-native pointer domain it is at most 30 bits wide, so it fits injectively in
/// the BabyBear field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, AlignedBorrow, StructReflection)]
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

impl<S, T: openvm_stark_backend::p3_field::PrimeCharacteristicRing> MemoryAddress<S, T> {
    /// Builds a bus address from little-endian 16-bit limbs of a block-aligned AS-native cell
    /// pointer. The caller must constrain the limbs to be canonical and the low limb to be
    /// divisible by [`BLOCK_FE_WIDTH`].
    #[inline(always)]
    pub fn from_cell_pointer_limbs<U: Into<T>>(
        address_space: S,
        [lo, hi]: [T; 2],
        block_width_inverse: U,
    ) -> Self {
        let pointer = lo * block_width_inverse.into()
            + hi * T::from_u32(1 << (16 - MEMORY_BLOCK_INDEX_SHIFT));
        Self {
            address_space,
            pointer,
        }
    }
}

impl<S: Clone, T: openvm_stark_backend::p3_field::PrimeCharacteristicRing> MemoryAddress<S, T> {
    /// Returns the address `blocks` memory-bus blocks after `self`.
    #[inline(always)]
    pub fn offset_blocks(&self, blocks: usize) -> Self {
        Self::new(
            self.address_space.clone(),
            self.pointer.clone() + T::from_usize(blocks),
        )
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
        let memory_dims = mem_config.memory_dimensions();
        let boundary = PersistentBoundaryAir::<VM_DIGEST_WIDTH> {
            memory_bus,
            merkle_bus,
            compression_bus,
        };
        let merkle = MemoryMerkleAir::<VM_DIGEST_WIDTH> {
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
