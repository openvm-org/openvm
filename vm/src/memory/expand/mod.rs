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

pub struct ExpandChip<const CHUNK: usize, F: PrimeField32> {
    pub air: ExpandAir<CHUNK>,
    initial_tree: MemoryNode<CHUNK, F>,
    touched_nodes: HashSet<(usize, usize, usize)>,
    num_touched_nonleaves: usize,
}

impl<const CHUNK: usize, F: PrimeField32> ExpandChip<CHUNK, F> {
    pub fn new(
        as_height: usize,
        address_height: usize,
        as_offset: usize,
        initial_tree: MemoryNode<CHUNK, F>,
    ) -> Self {
        assert!(as_height > 0);
        assert!(address_height > 0);
        let mut touched_nodes = HashSet::new();
        touched_nodes.insert((as_height + address_height, 0, 0));
        Self {
            air: ExpandAir {
                as_height,
                address_height,
                as_offset,
            },
            initial_tree,
            touched_nodes,
            num_touched_nonleaves: 1,
        }
    }

    fn touch_node(&mut self, height: usize, as_label: usize, address_label: usize) {
        println!("{} {} {}", height, as_label, address_label);
        if self.touched_nodes.insert((height, as_label, address_label)) {
            assert_ne!(height, self.air.as_height + self.air.address_height);
            if height != 0 {
                self.num_touched_nonleaves += 1;
            }
            self.touch_node(height + 1, as_label / 2, address_label / 2);
        }
    }

    pub fn touch_address(&mut self, address_space: F, address: F) {
        self.touch_node(
            0,
            ((address_space.as_canonical_u64() as usize) - self.air.as_offset)
                << self.air.address_height,
            (address.as_canonical_u64() as usize) / CHUNK,
        );
    }

    pub fn get_trace_height(&self) -> usize {
        2 * self.num_touched_nonleaves
    }
}
