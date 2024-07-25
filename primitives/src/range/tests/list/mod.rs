use std::sync::Arc;

use air::ListAir;

use crate::range::RangeCheckerChip;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Clone, Debug)]
pub struct ListChip {
    pub air: ListAir,
    pub vals: Vec<u32>,
    range_checker: Arc<RangeCheckerChip>,
}

impl ListChip {
    pub fn new(bus_index: usize, vals: Vec<u32>, range_checker: Arc<RangeCheckerChip>) -> Self {
        Self {
            air: ListAir { bus_index },
            vals,
            range_checker,
        }
    }

    pub fn range_max(&self) -> u32 {
        self.range_checker.air.range_max
    }

    pub fn bus_index(&self) -> usize {
        self.air.bus_index
    }
}
