pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

use crate::range::RangeCheckerChip;

use std::sync::Arc;

#[derive(Default)]
pub struct ListChip<const MAX: u32> {
    /// The index for the Range Checker bus.
    bus_index: usize,
    pub vals: Vec<u32>,

    range_checker: Arc<RangeCheckerChip<MAX>>,
}

impl<const MAX: u32> ListChip<MAX> {
    pub fn new(
        bus_index: usize,
        vals: Vec<u32>,
        range_checker: Arc<RangeCheckerChip<MAX>>,
    ) -> Self {
        Self {
            bus_index,
            vals,
            range_checker,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn vals(&self) -> &Vec<u32> {
        &self.vals
    }
}
