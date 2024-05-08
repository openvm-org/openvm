// Adapted from Valida

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

use std::collections::BTreeMap;

#[derive(Default)]
pub struct RangeCheckerChip<const MAX: u32> {
    /// The index for the Range Checker bus.
    bus_index: usize,
    pub count: BTreeMap<u32, u32>,
}

impl<const MAX: u32> RangeCheckerChip<MAX> {
    pub fn new(bus_index: usize) -> Self {
        Self {
            bus_index,
            count: BTreeMap::new(),
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }
}
