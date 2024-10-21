use p3_field::PrimeField32;

use crate::system::memory::{
    merkle::MemoryMerkleChip, persistent::PersistentBoundaryChip, volatile::VolatileBoundaryChip,
    Equipartition, CHUNK,
};

#[derive(Clone, Debug)]
pub enum MemoryInterface<F> {
    Volatile {
        boundary_chip: VolatileBoundaryChip<F>,
    },
    Persistent {
        boundary_chip: PersistentBoundaryChip<F, CHUNK>,
        merkle_chip: MemoryMerkleChip<CHUNK, F>,
        initial_memory: Equipartition<F, CHUNK>,
    },
}

impl<F: PrimeField32> MemoryInterface<F> {
    pub fn touch_address(&mut self, addr_space: F, address: F) {
        match self {
            MemoryInterface::Volatile { boundary_chip } => {
                boundary_chip.touch_address(addr_space, address);
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => {
                boundary_chip.touch_address(addr_space, address);
                merkle_chip.touch_address(addr_space, address);
            }
        }
    }
}
