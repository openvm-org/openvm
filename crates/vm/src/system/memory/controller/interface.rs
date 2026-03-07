use openvm_stark_backend::{interaction::PermutationCheckBus, p3_field::PrimeField32};

use crate::system::memory::{
    merkle::{MemoryMerkleAir, MemoryMerkleChip},
    persistent::{PersistentBoundaryAir, PersistentBoundaryChip},
    MemoryImage, CHUNK,
};

#[derive(Clone)]
pub struct MemoryInterfaceAirs {
    pub boundary: PersistentBoundaryAir<CHUNK>,
    pub merkle: MemoryMerkleAir<CHUNK>,
}

pub struct MemoryInterface<F> {
    pub boundary_chip: PersistentBoundaryChip<F, CHUNK>,
    pub merkle_chip: MemoryMerkleChip<CHUNK, F>,
    pub initial_memory: MemoryImage,
}

impl<F: PrimeField32> MemoryInterface<F> {
    pub fn compression_bus(&self) -> PermutationCheckBus {
        self.merkle_chip.air.compression_bus
    }
}
