// Adapted from Valida

use std::sync::{atomic::AtomicU32, Arc};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub use air::RangeCheckerAir;

#[derive(Clone, Default, Debug)]
pub struct RangeCheckerChip {
    pub air: RangeCheckerAir,
    count: Vec<Arc<AtomicU32>>,
}

impl RangeCheckerChip {
    pub fn new(bus_index: usize, range_max: u32) -> Self {
        let mut count = vec![];
        for _ in 0..range_max {
            count.push(Arc::new(AtomicU32::new(0)));
        }

        Self {
            air: RangeCheckerAir {
                bus_index,
                range_max,
            },
            count,
        }
    }

    pub fn add_count(&self, val: u32) {
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
