use std::sync::Arc;

use air::ListAir;
use openvm_stark_backend::interaction::BusIndex;

use crate::range::RangeCheckerChip;

pub mod air;
pub mod columns;
pub mod trace;

pub struct ListChip {
    pub air: ListAir,
    pub vals: Vec<u32>,
    range_checker: Arc<RangeCheckerChip>,
}

impl ListChip {
    pub fn new(vals: Vec<u32>, range_checker: Arc<RangeCheckerChip>) -> Self {
        let bus = range_checker.bus();
        Self {
            air: ListAir::new(bus),
            vals,
            range_checker,
        }
    }

    pub fn range_max(&self) -> u32 {
        self.range_checker.range_max()
    }

    pub fn bus_index(&self) -> BusIndex {
        self.air.bus.inner.index
    }
}
