use openvm_stark_backend::{interaction::PermutationCheckBus, p3_field::PrimeField32};

use crate::system::memory::{
    merkle::MemoryMerkleChip, persistent::PersistentBoundaryChip, volatile::VolatileBoundaryChip,
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
