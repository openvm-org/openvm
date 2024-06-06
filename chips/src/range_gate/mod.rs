use std::sync::{atomic::AtomicU32, Arc};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct RangeCheckerGateAir {
    bus_index: usize,
    _range_max: u32,
}

/// This chip gets requests to verify that a number is in the range
/// [0, MAX). In the trace, there is a counter column and a multiplicity
/// column. The counter column is generated using a gate, as opposed to
/// the other RangeCheckerChip.
#[derive(Default)]
pub struct RangeCheckerGateChip {
    pub air: RangeCheckerGateAir,
    pub count: Vec<Arc<AtomicU32>>,
}

impl RangeCheckerGateChip {
    pub fn new(bus_index: usize, range_max: u32) -> Self {
        let mut count = vec![];
        for _ in 0..range_max {
            count.push(Arc::new(AtomicU32::new(0)));
        }
        Self {
            air: RangeCheckerGateAir {
                bus_index,
                _range_max: range_max,
            },
            count,
        }
    }

    pub fn add_count(&self, val: u32) {
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
