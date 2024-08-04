use std::collections::{HashMap, HashSet};

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
    pub height: usize,
    initial_trees: HashMap<F, MemoryNode<CHUNK, F>>,
    touched_nodes: HashSet<(F, usize, usize)>,
    num_touched_nonleaves: usize,
}

impl<const CHUNK: usize, F: PrimeField32> ExpandChip<CHUNK, F> {
    pub fn new(height: usize, initial_trees: HashMap<F, MemoryNode<CHUNK, F>>) -> Self {
        let touched_nodes = initial_trees
            .keys()
            .map(|&address_space| (address_space, height, 0))
            .collect();
        let num_touched_nonleaves = initial_trees.len();
        Self {
            height,
            initial_trees,
            touched_nodes,
            num_touched_nonleaves,
        }
    }

    pub fn air(&self) -> ExpandAir<CHUNK> {
        ExpandAir {}
    }

    fn touch_node(&mut self, address_space: F, height: usize, label: usize) {
        if self.touched_nodes.insert((address_space, height, label)) {
            assert_ne!(height, self.height);
            if height != 0 {
                self.num_touched_nonleaves += 1;
            }
            self.touch_node(address_space, height + 1, label / 2);
        }
    }

    pub fn touch_address(&mut self, address_space: F, address: F) {
        self.touch_node(
            address_space,
            0,
            (address.as_canonical_u64() as usize) / CHUNK,
        );
    }

    pub fn get_trace_height(&self) -> usize {
        2 * self.num_touched_nonleaves
    }
}
