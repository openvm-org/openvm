use std::{collections::HashSet, marker::PhantomData};

use p3_field::PrimeField32;

use super::manager::dimensions::MemoryDimensions;
mod air;
mod bridge;
mod columns;
mod trace;

pub use air::*;
pub use bridge::*;
pub use columns::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct MemoryMerkleChip<const CHUNK: usize, F> {
    pub air: MemoryMerkleAir<CHUNK>,
    touched_nodes: HashSet<(usize, u64, usize)>,
    num_touched_nonleaves: usize,
    _marker: PhantomData<F>,
}

impl<const CHUNK: usize, F: PrimeField32> MemoryMerkleChip<CHUNK, F> {
    pub fn new(memory_dimensions: MemoryDimensions, merkle_bus: MemoryMerkleBus) -> Self {
        assert!(memory_dimensions.as_height > 0);
        assert!(memory_dimensions.address_height > 0);
        let mut touched_nodes = HashSet::new();
        touched_nodes.insert((memory_dimensions.overall_height(), 0, 0));
        Self {
            air: MemoryMerkleAir {
                memory_dimensions,
                merkle_bus,
            },
            touched_nodes,
            num_touched_nonleaves: 1,
            _marker: PhantomData,
        }
    }

    fn touch_node(&mut self, height: usize, as_label: u64, address_label: usize) {
        if self.touched_nodes.insert((height, as_label, address_label)) {
            assert_ne!(height, self.air.memory_dimensions.overall_height());
            if height != 0 {
                self.num_touched_nonleaves += 1;
            }
            self.touch_node(height + 1, as_label / 2, address_label / 2);
        }
    }

    pub fn touch_address(&mut self, address_space: F, address: F) {
        let as_label = self.air.memory_dimensions.as_to_as_label(address_space);
        self.touch_node(
            0,
            as_label,
            ((address.as_canonical_u32() as usize) / CHUNK) * 2,
        );
        self.touch_node(
            0,
            as_label,
            ((address.as_canonical_u32() as usize) / CHUNK) * 2 + 1,
        );
    }

    pub fn current_height(&self) -> usize {
        2 * self.num_touched_nonleaves
    }
}
