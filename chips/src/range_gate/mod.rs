use std::sync::{atomic::AtomicU32, Arc};

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

/// This chip gets requests to verify that a number is in the range
/// [0, MAX). In the trace, there is a counter column and a multiplicity
/// column. The counter column is generated using a gate, as opposed to
/// the other RangeCheckerChip.
#[derive(Default)]
pub struct RangeCheckerGateChip<const MAX: u32> {
    /// The index for the Range Checker bus.
    bus_index: usize,
    pub count: Vec<Arc<AtomicU32>>,
}

impl<const MAX: u32> RangeCheckerGateChip<MAX> {
    pub fn new(bus_index: usize) -> Self {
        let mut count = vec![];
        for _ in 0..MAX {
            count.push(Arc::new(AtomicU32::new(0)));
        }
        Self { bus_index, count }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn add_count(&self, val: u32) {
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
