use std::collections::{HashMap, HashSet};

use p3_field::{Field, PrimeField32};

use crate::memory::{expand::MemoryDimensions, interface::air::MemoryInterfaceAir};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

pub const EXPAND_BUS: usize = 4;
pub const MEMORY_INTERFACE_BUS: usize = 5;

pub struct AccessCell<const WORD_SIZE: usize, F: Field> {
    value: [F; WORD_SIZE],
    timestamp: F,
}

pub struct MemoryInterfaceChip<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> {
    pub air: MemoryInterfaceAir<NUM_WORDS, WORD_SIZE>,
    touched_leaves: HashSet<(F, usize)>,
    initial_memory: HashMap<(F, F), AccessCell<WORD_SIZE, F>>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    MemoryInterfaceChip<NUM_WORDS, WORD_SIZE, F>
{
    pub fn new(memory_dimensions: MemoryDimensions) -> Self {
        Self {
            air: MemoryInterfaceAir { memory_dimensions },
            touched_leaves: HashSet::new(),
            initial_memory: HashMap::new(),
        }
    }
    pub fn touch_address(
        &mut self,
        address_space: F,
        address: F,
        old_value: [F; WORD_SIZE],
        timestamp: F,
    ) {
        let leaf_label = (address.as_canonical_u64() as usize) / (NUM_WORDS * WORD_SIZE);
        self.touched_leaves.insert((address_space, leaf_label));
        self.initial_memory
            .entry((address_space, address))
            .or_insert_with(|| AccessCell {
                value: old_value,
                timestamp,
            });
    }

    pub fn get_trace_height(&self) -> usize {
        2 * self.touched_leaves.len()
    }
}
