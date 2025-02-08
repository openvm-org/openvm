use openvm_stark_backend::p3_field::PrimeField32;

use crate::system::memory::{
    merkle::{DirectCompressionBus, MemoryMerkleChip},
    persistent::PersistentBoundaryChip,
    volatile::VolatileBoundaryChip,
    MemoryImage, CHUNK,
};

#[allow(clippy::large_enum_variant)]
pub enum MemoryInterface<F> {
    Volatile {
        boundary_chip: VolatileBoundaryChip<F>,
    },
    Persistent {
        boundary_chip: PersistentBoundaryChip<F, CHUNK>,
        merkle_chip: MemoryMerkleChip<CHUNK, F>,
        initial_memory: MemoryImage<F>,
    },
}

impl<F: PrimeField32> MemoryInterface<F> {
    pub fn touch_range(&mut self, addr_space: u32, pointer: u32, len: u32) {
        match self {
            Self::Volatile { .. } => {}
            Self::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => {
                boundary_chip.touch_range(addr_space, pointer, len);
                merkle_chip.touch_range(addr_space, pointer, len);
            }
        }
    }

    pub fn compression_bus(&self) -> Option<DirectCompressionBus> {
        match self {
            Self::Volatile { .. } => None,
            Self::Persistent { merkle_chip, .. } => {
                Some(merkle_chip.air.compression_bus)
            }
        }
    }
}
