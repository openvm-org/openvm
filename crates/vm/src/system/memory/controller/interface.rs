use openvm_stark_backend::{interaction::PermutationCheckBus, p3_field::PrimeField32};

use crate::system::memory::{
    merkle::{MemoryMerkleAir, MemoryMerkleChip},
    persistent::{PersistentBoundaryAir, PersistentBoundaryChip},
    volatile::{VolatileBoundaryAir, VolatileBoundaryChip},
    MemoryImage, CHUNK,
};

/// Block size for volatile memory boundary AIR.
/// Uses CHUNK (e.g. 8), and the AIR will split into CONST_BLOCK_SIZE segments on the bus.
pub const VOLATILE_BOUNDARY_BLOCK_SIZE: usize = CHUNK;

#[derive(Clone)]
pub enum MemoryInterfaceAirs {
    Volatile {
        boundary: VolatileBoundaryAir<VOLATILE_BOUNDARY_BLOCK_SIZE>,
    },
    Persistent {
        boundary: PersistentBoundaryAir<CHUNK>,
        merkle: MemoryMerkleAir<CHUNK>,
    },
}

#[allow(clippy::large_enum_variant)]
pub enum MemoryInterface<F> {
    Volatile {
        boundary_chip: VolatileBoundaryChip<F, VOLATILE_BOUNDARY_BLOCK_SIZE>,
    },
    Persistent {
        boundary_chip: PersistentBoundaryChip<F, CHUNK>,
        merkle_chip: MemoryMerkleChip<CHUNK, F>,
        initial_memory: MemoryImage,
    },
}

impl<F: PrimeField32> MemoryInterface<F> {
    pub fn compression_bus(&self) -> Option<PermutationCheckBus> {
        match self {
            MemoryInterface::Volatile { .. } => None,
            MemoryInterface::Persistent { merkle_chip, .. } => {
                Some(merkle_chip.air.compression_bus)
            }
        }
    }
}
