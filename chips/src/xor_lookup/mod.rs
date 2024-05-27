pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

use std::sync::atomic::AtomicU32;

/// This chip gets requests to compute the xor of two numbers x and y of at most M bits.
/// It generates a preprocessed table with a row for each possible triple (x, y, x^y)
/// and keeps count of the number of times each triple is requested for the single main trace column.
#[derive(Default)]
pub struct XorLookupChip<const M: usize> {
    bus_index: usize,
    pub count: Vec<Vec<AtomicU32>>,
}

impl<const M: usize> XorLookupChip<M> {
    pub fn new(bus_index: usize) -> Self {
        let mut count = vec![];
        for _ in 0..(1 << M) {
            let mut row = vec![];
            for _ in 0..(1 << M) {
                row.push(AtomicU32::new(0));
            }
            count.push(row);
        }
        Self { bus_index, count }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    fn calc_xor(&self, x: u32, y: u32) -> u32 {
        x ^ y
    }

    pub fn request(&self, x: u32, y: u32) -> u32 {
        let val_atomic = &self.count[x as usize][y as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        self.calc_xor(x, y)
    }
}
