use std::{collections::HashSet, sync::Arc};

use afs_primitives::var_range::VariableRangeCheckerChip;
use p3_field::PrimeField32;

use self::air::VolatileBoundaryAir;
use crate::system::memory::offline_checker::MemoryBus;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct VolatileBoundaryChip<F> {
    pub air: VolatileBoundaryAir,
    touched_addresses: HashSet<(F, F)>,
    range_checker: Arc<VariableRangeCheckerChip>,
}

impl<F: PrimeField32> VolatileBoundaryChip<F> {
    pub fn new(
        memory_bus: MemoryBus,
        addr_space_max_bits: usize,
        address_max_bits: usize,
        decomp: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        Self {
            air: VolatileBoundaryAir::new(
                memory_bus,
                addr_space_max_bits,
                address_max_bits,
                decomp,
                false,
            ),
            touched_addresses: HashSet::new(),
            range_checker,
        }
    }

    pub fn touch_address(&mut self, addr_space: F, address: F) {
        self.touched_addresses.insert((addr_space, address));
    }

    pub fn all_addresses(&self) -> Vec<(F, F)> {
        self.touched_addresses.iter().cloned().collect()
    }

    pub fn current_height(&self) -> usize {
        self.touched_addresses.len()
    }
}
