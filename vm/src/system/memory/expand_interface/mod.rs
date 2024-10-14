use std::collections::{HashMap, HashSet};

use p3_field::PrimeField32;

use self::air::MemoryExpandInterfaceAir;
use super::manager::{access_cell::AccessCell, dimensions::MemoryDimensions};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

pub struct MemoryExpandInterfaceChip<
    const NUM_WORDS: usize,
    const WORD_SIZE: usize,
    F: PrimeField32,
> {
    pub air: MemoryExpandInterfaceAir<NUM_WORDS, WORD_SIZE>,
    touched_leaves: HashSet<(F, usize)>,
    initial_memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryExpandInterfaceChip<NUM_WORDS, WORD_SIZE, F>
{
    pub fn new(memory_dimensions: MemoryDimensions) -> Self {
        Self {
            air: MemoryExpandInterfaceAir { memory_dimensions },
            touched_leaves: HashSet::new(),
            initial_memory: HashMap::new(),
        }
    }
    pub fn touch_address(&mut self, addr_space: F, pointer: F, old_value: [F; WORD_SIZE], clk: F) {
        let leaf_label = (pointer.as_canonical_u64() as usize) / (NUM_WORDS * WORD_SIZE);
        self.touched_leaves.insert((addr_space, leaf_label));
        self.initial_memory
            .entry((addr_space, pointer))
            .or_insert_with(|| AccessCell {
                data: old_value,
                clk,
            });
    }

    pub fn all_addresses(&self) -> Vec<(F, F)> {
        self.initial_memory.keys().cloned().collect()
    }

    pub fn get_trace_height(&self) -> usize {
        2 * self.touched_leaves.len()
    }
}
