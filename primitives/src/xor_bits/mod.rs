use std::sync::Arc;

use air::XorBitsAir;
use parking_lot::Mutex;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

#[derive(Clone, Debug)]
/// A chip that computes the xor of two numbers of at most N bits each.
/// This chip consists of the AIR as well as a receiver to handle counting requests.
pub struct XorBitsChip<const N: usize> {
    pub air: XorBitsAir<N>,

    /// List of all requests sent to the chip
    pairs: Arc<Mutex<Vec<(u32, u32)>>>,
}

impl<const N: usize> XorBitsChip<N> {
    pub fn new(bus_index: usize, pairs: Vec<(u32, u32)>) -> Self {
        Self {
            air: XorBitsAir { bus_index },
            pairs: Arc::new(Mutex::new(pairs)),
        }
    }

    pub fn request(&self, a: u32, b: u32) -> u32 {
        let mut pairs_locked = self.pairs.lock();
        pairs_locked.push((a, b));
        self.air.calc_xor(a, b)
    }
}
