// Adapted from Valida

use std::sync::{atomic::AtomicU32, Arc};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

pub use air::RangeTupleCheckerAir;

#[derive(Clone, Default, Debug)]
pub struct RangeTupleCheckerChip {
    pub air: RangeTupleCheckerAir,
    sizes: Vec<u32>,
    count: Vec<Arc<AtomicU32>>,
}

impl RangeTupleCheckerChip {
    pub fn new(bus_index: usize, sizes: Vec<u32>) -> Self {
        let mut count = vec![];
        let range_max = sizes.iter().product();
        for _ in 0..range_max {
            count.push(Arc::new(AtomicU32::new(0)));
        }

        Self {
            air: RangeTupleCheckerAir {
                bus_index,
                sizes: sizes.clone(),
            },
            sizes,
            count,
        }
    }

    pub fn add_count(&self, ids: &[u32]) {
        let index = ids
            .iter()
            .zip(self.sizes.iter())
            .fold(0, |acc, (id, sz)| acc * sz + id);
        let val_atomic = &self.count[index as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
