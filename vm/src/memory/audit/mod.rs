use std::{collections::BTreeMap, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use p3_field::PrimeField32;

use self::air::MemoryAuditAir;
use super::manager::access_cell::AccessCell;
use crate::memory::offline_checker::bus::MemoryBus;

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
            air: MemoryAuditAir::new(
                memory_bus,
                addr_space_max_bits,
                pointer_max_bits,
                decomp,
                false,
            ),
            initial_memory: BTreeMap::new(),
            range_checker,
        }
    }

    pub fn new_for_testing(
        memory_bus: MemoryBus,
        addr_space_max_bits: usize,
        pointer_max_bits: usize,
        decomp: usize,
    ) -> Self {
        Self {
            air: MemoryAuditAir::new(
                memory_bus,
                addr_space_max_bits,
                pointer_max_bits,
                decomp,
                true,
            ),
            initial_memory: BTreeMap::new(),
            range_checker: Arc::new(RangeCheckerGateChip::new(0, 1 << decomp)),
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
