use std::sync::atomic::AtomicU32;

pub mod air;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

pub use air::RangeCheckerGateAir;

use crate::range::bus::RangeCheckBus;

/// This chip gets requests to verify that a number is in the range
/// [0, MAX). In the trace, there is a counter column and a multiplicity
/// column. The counter column is generated using a gate, as opposed to
/// the other RangeCheckerChip.
#[derive(Debug)]
pub struct RangeCheckerGateChip {
    pub air: RangeCheckerGateAir,
    pub count: Vec<AtomicU32>,
}

impl RangeCheckerGateChip {
    pub fn new(bus: RangeCheckBus) -> Self {
        let count = (0..bus.range_max).map(|_| AtomicU32::new(0)).collect();

        Self {
            air: RangeCheckerGateAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> RangeCheckBus {
        self.air.bus
    }

    pub fn bus_index(&self) -> usize {
        self.air.bus.index
    }

    pub fn range_max(&self) -> u32 {
        self.air.bus.range_max
    }

    pub fn air_width(&self) -> usize {
        2
    }

    pub fn add_count(&self, val: u32) {
        assert!(
            val < self.range_max(),
            "range exceeded: {} >= {}",
            val,
            self.range_max()
        );
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for i in 0..self.count.len() {
            self.count[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
