use openvm_stark_backend::{interaction::PermutationCheckBus, p3_field::PrimeField32};

use crate::system::memory::{
    boundary::{MemoryBoundaryAir, MemoryBoundaryChip},
    merkle::{MemoryMerkleAir, MemoryMerkleChip},
    MemoryImage, CHUNK,
};

#[derive(Clone)]
pub struct MemoryInterfaceAirs {
    pub boundary: MemoryBoundaryAir<CHUNK>,
    pub merkle: MemoryMerkleAir<CHUNK>,
}

pub struct MemoryInterface<F> {
    pub boundary_chip: MemoryBoundaryChip<F, CHUNK>,
    pub merkle_chip: MemoryMerkleChip<CHUNK, F>,
    pub initial_memory: MemoryImage,
}

impl<F: PrimeField32> MemoryInterface<F> {
    pub fn compression_bus(&self) -> PermutationCheckBus {
        self.merkle_chip.air.compression_bus
    }
}
