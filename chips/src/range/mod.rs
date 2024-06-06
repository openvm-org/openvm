// Adapted from Valida

use std::sync::{atomic::AtomicU32, Arc};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct RangeCheckerAir {
    pub bus_index: usize,
    pub range_max: u32,
}

#[derive(Default)]
pub struct RangeCheckerChip {
    pub air: RangeCheckerAir,
    pub count: Vec<Arc<AtomicU32>>,
}

impl RangeCheckerChip {
    pub fn new(bus_index: usize, range_max: u32) -> Self {
        let count = (0..range_max)
            .map(|_| Arc::new(AtomicU32::new(0)))
            .collect();

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
