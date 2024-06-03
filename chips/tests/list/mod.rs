use std::sync::Arc;

use crate::range::RangeCheckerChip;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct ListChip {
    /// The index for the Range Checker bus.
    bus_index: usize,
    _range_max: u32,
    pub vals: Vec<u32>,

    range_checker: Arc<RangeCheckerChip>,
}

impl ListChip {
    pub fn new(
        bus_index: usize,
        range_max: u32,
        vals: Vec<u32>,
        range_checker: Arc<RangeCheckerChip>,
    ) -> Self {
        Self {
            bus_index,
            _range_max: range_max,
            vals,
            range_checker,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }
}
