use std::sync::atomic::AtomicU32;

pub mod air;
pub mod bus;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub use air::VariableRangeCheckerAir;
use bus::VariableRangeCheckerBus;

#[derive(Debug)]
pub struct VariableRangeCheckerChip {
    pub air: VariableRangeCheckerAir,
    count: Vec<AtomicU32>,
}

impl VariableRangeCheckerChip {
    pub fn new(bus: VariableRangeCheckerBus) -> Self {
        let num_rows = (1 << (bus.range_max_bits + 1)) as usize;
        let count = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        Self {
            air: VariableRangeCheckerAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> VariableRangeCheckerBus {
        self.air.bus
    }

    pub fn range_max_bits(&self) -> usize {
        self.air.range_max_bits()
    }

    pub fn add_count(&self, value: u32, max_bits: usize) {
        // index is 2^max_bits + value - 1 + 1 for the extra [0, 0] row
        // if each [value, max_bits] is valid, the sends multiset will be exactly the receives multiset
        let idx = (1 << max_bits) + (value as usize);
        assert!(
            idx < self.count.len(),
            "range exceeded: {} >= {}",
            idx,
            self.count.len()
        );
        let val_atomic = &self.count[idx];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
