use std::collections::HashSet;

use p3_field::PrimeField32;

use crate::memory::{expand::air::ExpandAir, tree::MemoryNode};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

pub const EXPAND_BUS: usize = 4;
pub const POSEIDON2_DIRECT_REQUEST_BUS: usize = 6;

// indicates that there are 2^`as_height` address spaces numbered starting from `as_offset`,
// and that each address space has 2^`address_height` addresses numbered starting from 0
#[derive(Clone, Copy)]
pub struct MemoryDimensions {
    pub as_height: usize,
    pub address_height: usize,
    pub as_offset: usize,
}

impl MemoryDimensions {
    pub fn overall_height(&self) -> usize {
        self.as_height + self.address_height
    }
}

pub struct ExpandChip<const CHUNK: usize, F: PrimeField32> {
    pub air: ExpandAir<CHUNK>,
    initial_tree: MemoryNode<CHUNK, F>,
    touched_nodes: HashSet<(usize, usize, usize)>,
    num_touched_nonleaves: usize,
}

impl<const CHUNK: usize, F: PrimeField32> ExpandChip<CHUNK, F> {
    pub fn new(memory_dimensions: MemoryDimensions, initial_tree: MemoryNode<CHUNK, F>) -> Self {
        assert!(memory_dimensions.as_height > 0);
        assert!(memory_dimensions.address_height > 0);
        let mut touched_nodes = HashSet::new();
        touched_nodes.insert((memory_dimensions.overall_height(), 0, 0));
        Self {
            air: ExpandAir { memory_dimensions },
            initial_tree,
            touched_nodes,
            num_touched_nonleaves: 1,
        }
    }

    fn touch_node(&mut self, height: usize, as_label: usize, address_label: usize) {
        println!("{} {} {}", height, as_label, address_label);
        if self.touched_nodes.insert((height, as_label, address_label)) {
            assert_ne!(height, self.air.memory_dimensions.overall_height());
            if height != 0 {
                self.num_touched_nonleaves += 1;
            }
            self.touch_node(height + 1, as_label / 2, address_label / 2);
        }
    }

    pub fn touch_address(&mut self, address_space: F, address: F) {
        self.touch_node(
            0,
            ((address_space.as_canonical_u64() as usize) - self.air.memory_dimensions.as_offset)
                << self.air.memory_dimensions.address_height,
            (address.as_canonical_u64() as usize) / CHUNK,
        );
    }

    pub fn get_trace_height(&self) -> usize {
        2 * self.num_touched_nonleaves
    }
}
