use std::{collections::HashMap, sync::Arc};

use afs_primitives::var_range::VariableRangeCheckerChip;
use p3_field::Field;

use self::air::MemoryAuditAir;
use crate::system::memory::offline_checker::MemoryBus;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct MemoryAuditChip<F: Field> {
    pub air: MemoryAuditAir,
    initial_memory: HashMap<(F, F), F>,
    range_checker: Arc<VariableRangeCheckerChip>,
}

impl<F: Field> MemoryAuditChip<F> {
    pub fn new(
        memory_bus: MemoryBus,
        addr_space_max_bits: usize,
        pointer_max_bits: usize,
        decomp: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        Self {
            air: MemoryAuditAir::new(
                memory_bus,
                addr_space_max_bits,
                pointer_max_bits,
                decomp,
                false,
            ),
            initial_memory: HashMap::new(),
            range_checker,
        }
    }

    pub fn touch_address(&mut self, addr_space: F, pointer: F, old_data: F) {
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
