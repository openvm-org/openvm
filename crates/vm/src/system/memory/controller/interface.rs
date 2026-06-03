use openvm_stark_backend::{interaction::PermutationCheckBus, p3_field::PrimeField32};

use crate::system::memory::{
    merkle::{MemoryMerkleAir, MemoryMerkleChip},
    persistent::{PersistentBoundaryAir, PersistentBoundaryChip},
    MemoryImage, DIGEST_WIDTH,
};

#[derive(Clone)]
pub struct MemoryInterfaceAirs {
    pub boundary: PersistentBoundaryAir<DIGEST_WIDTH>,
    pub merkle: MemoryMerkleAir<DIGEST_WIDTH>,
}

pub struct MemoryInterface<F> {
    pub boundary_chip: PersistentBoundaryChip<F, DIGEST_WIDTH>,
    pub merkle_chip: MemoryMerkleChip<DIGEST_WIDTH, F>,
    pub initial_memory: MemoryImage,
}

impl<F: PrimeField32> MemoryInterface<F> {
    pub fn compression_bus(&self) -> PermutationCheckBus {
        self.merkle_chip.air.compression_bus
    }
}
