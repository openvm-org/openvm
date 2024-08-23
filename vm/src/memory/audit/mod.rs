use std::{collections::BTreeMap, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use p3_field::PrimeField32;

use self::air::MemoryAuditAir;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct MemoryAuditChip<const WORD_SIZE: usize, F: PrimeField32> {
    pub air: MemoryAuditAir<WORD_SIZE>,
    initial_memory: BTreeMap<(F, F), [F; WORD_SIZE]>,
    range_checker: Arc<RangeCheckerGateChip>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> MemoryAuditChip<WORD_SIZE, F> {
    pub fn new(
        addr_space_max_bits: usize,
        pointer_max_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: MemoryAuditAir::new(addr_space_max_bits, pointer_max_bits, decomp),
            initial_memory: BTreeMap::new(),
            range_checker,
        }
    }

    pub fn touch_address(&mut self, addr_space: F, pointer: F, old_data: [F; WORD_SIZE]) {
        self.initial_memory
            .entry((addr_space, pointer))
            .or_insert(old_data);
    }

    pub fn all_addresses(&self) -> Vec<(F, F)> {
        self.initial_memory.keys().cloned().collect()
    }

    pub fn current_height(&self) -> usize {
        self.initial_memory.len()
    }
}
