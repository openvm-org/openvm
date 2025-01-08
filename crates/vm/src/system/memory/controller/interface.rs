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
    pub fn touch_address(&mut self, addr_space: u32, pointer: u32) {
        match self {
            MemoryInterface::Volatile { boundary_chip } => {
                boundary_chip.touch_address(addr_space, pointer);
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => {
                boundary_chip.touch_address(addr_space, pointer);
                merkle_chip.touch_address(addr_space, pointer);
            }
        }
    }

    pub fn touch_range(&mut self, addr_space: u32, pointer: u32, len: u32) {
        match self {
            MemoryInterface::Volatile { boundary_chip } => {
                for offset in 0..len {
                    boundary_chip.touch_address(addr_space, pointer + offset);
                }
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => {
                for offset in 0..len {
                    boundary_chip.touch_address(addr_space, pointer + offset);
                    merkle_chip.touch_address(addr_space, pointer + offset);
                }
            }
        }
    }

    pub fn compression_bus(&self) -> Option<DirectCompressionBus> {
        match self {
            MemoryInterface::Volatile { .. } => None,
            MemoryInterface::Persistent { merkle_chip, .. } => {
                Some(merkle_chip.air.compression_bus)
            }
        }
    }
}
