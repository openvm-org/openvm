use std::{collections::BTreeMap, sync::Arc};

use p3_field::PrimeField32;

use afs_primitives::range_gate::RangeCheckerGateChip;

use crate::memory::offline_checker::bus::MemoryBus;

use super::manager::access_cell::AccessCell;

use self::air::MemoryAuditAir;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct MemoryAuditChip<const WORD_SIZE: usize, F: PrimeField32> {
    pub air: MemoryAuditAir<WORD_SIZE>,
    initial_memory: BTreeMap<(F, F), AccessCell<WORD_SIZE, F>>,
    range_checker: Arc<RangeCheckerGateChip>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> MemoryAuditChip<WORD_SIZE, F> {
    pub fn new(
        memory_bus: MemoryBus,
        addr_space_max_bits: usize,
        pointer_max_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: MemoryAuditAir::new(memory_bus, addr_space_max_bits, pointer_max_bits, decomp),
            initial_memory: BTreeMap::new(),
            range_checker,
        }
    }

    pub fn touch_address(&mut self, addr_space: F, pointer: F, old_data: [F; WORD_SIZE], clk: F) {
        self.initial_memory
            .entry((addr_space, pointer))
            .or_insert_with(|| AccessCell {
                data: old_data,
                clk,
            });
    }

    pub fn all_addresses(&self) -> Vec<(F, F)> {
        self.initial_memory.keys().cloned().collect()
    }

    pub fn current_height(&self) -> usize {
        self.initial_memory.len()
    }
}
