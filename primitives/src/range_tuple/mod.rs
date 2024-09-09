// Adapted from Valida

use std::sync::{atomic::AtomicU32, Arc};

pub mod air;
pub mod bus;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub use air::RangeTupleCheckerAir;
use bus::RangeTupleCheckerBus;

#[derive(Clone, Default, Debug)]
pub struct RangeTupleCheckerChip {
    pub air: RangeTupleCheckerAir,
    count: Vec<Arc<AtomicU32>>,
}

impl RangeTupleCheckerChip {
    pub fn new(bus: RangeTupleCheckerBus) -> Self {
        let range_max = bus.sizes.iter().product();
        let count = (0..range_max)
            .map(|_| Arc::new(AtomicU32::new(0)))
            .collect();

        Self {
            air: RangeTupleCheckerAir { bus },
            count,
        }
    }

    pub fn bus(&self) -> &RangeTupleCheckerBus {
        &self.air.bus
    }

    pub fn sizes(&self) -> &Vec<u32> {
        &self.air.bus.sizes
    }

    pub fn add_count(&self, ids: &[u32]) {
        let index = ids
            .iter()
            .zip(self.air.bus.sizes.iter())
            .fold(0, |acc, (id, sz)| acc * sz + id) as usize;
        assert!(
            index < self.count.len(),
            "range exceeded: {} >= {}",
            index,
            self.count.len()
        );
        let val_atomic = &self.count[index];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
