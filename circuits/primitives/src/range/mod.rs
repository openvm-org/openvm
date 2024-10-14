// Adapted from Valida

use std::sync::atomic::AtomicU32;

pub mod air;
pub mod bus;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub use air::RangeCheckerAir;
use bus::RangeCheckBus;

#[derive(Debug)]
pub struct RangeCheckerChip {
    pub air: RangeCheckerAir,
    count: Vec<AtomicU32>,
}

impl RangeCheckerChip {
    pub fn new(bus: RangeCheckBus) -> Self {
        let mut count = vec![];
        for _ in 0..bus.range_max {
            count.push(AtomicU32::new(0));
        }

        Self {
            air: RangeCheckerAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> RangeCheckBus {
        self.air.bus
    }

    pub fn range_max(&self) -> u32 {
        self.air.range_max()
    }

    pub fn add_count(&self, val: u32) {
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
