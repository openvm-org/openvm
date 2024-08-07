use std::collections::{HashMap, HashSet};

use p3_field::{Field, PrimeField32};

use crate::memory::{interface::air::MemoryInterfaceAir, OpType, OpType::Read};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

pub const EXPAND_BUS: usize = 4;
pub const MEMORY_INTERFACE_BUS: usize = 5;

struct Cell<F: Field> {
    read_initially: bool,
    initial_value: F,
}

#[derive(Default)]
pub struct MemoryInterfaceChip<const CHUNK: usize, F: PrimeField32> {
    touched_leaves: HashSet<(F, usize)>,
    touched_addresses: HashMap<(F, F), Cell<F>>,
}

impl<const CHUNK: usize, F: PrimeField32> MemoryInterfaceChip<CHUNK, F> {
    pub fn air(&self) -> MemoryInterfaceAir<CHUNK> {
        MemoryInterfaceAir {}
    }
    pub fn touch_address(&mut self, address_space: F, address: F, op_type: OpType, old_value: F) {
        let leaf_label = (address.as_canonical_u64() as usize) / CHUNK;
        self.touched_leaves.insert((address_space, leaf_label));
        self.touched_addresses
            .entry((address_space, address))
            .or_insert_with(|| Cell {
                read_initially: op_type == Read,
                initial_value: old_value,
            });
    }

    pub fn get_trace_height(&self) -> usize {
        2 * self.touched_leaves.len()
    }
}
