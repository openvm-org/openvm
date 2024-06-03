// Adapted from Valida

use std::sync::{atomic::AtomicU32, Arc};

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct RangeCheckerChip {
    /// The index for the Range Checker bus.
    bus_index: usize,
    range_max: u32,
    pub count: Vec<Arc<AtomicU32>>,
}

impl RangeCheckerChip {
    pub fn new(bus_index: usize, range_max: u32) -> Self {
        let mut count = vec![];
        for _ in 0..range_max {
            count.push(Arc::new(AtomicU32::new(0)));
        }
        Self {
            bus_index,
            range_max,
            count,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn add_count(&self, val: u32) {
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
