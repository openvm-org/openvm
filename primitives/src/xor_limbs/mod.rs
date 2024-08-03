use std::sync::Arc;

use air::XorLimbsAir;
use parking_lot::Mutex;

use crate::xor_lookup::XorLookupChip;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

/// This chip gets requests to compute the xor of two numbers x and y of at most N bits.
/// It breaks down those numbers into limbs of at most M bits each, and computes the xor of
/// those limbs by communicating with the `XorLookupChip`.
#[derive(Debug)]
pub struct XorLimbsChip<const N: usize, const M: usize> {
    pub air: XorLimbsAir<N, M>,

    pairs: Arc<Mutex<Vec<(u32, u32)>>>,
    pub xor_lookup_chip: XorLookupChip<M>,
}

impl<const N: usize, const M: usize> XorLimbsChip<N, M> {
    pub fn new(bus_index: usize, pairs: Vec<(u32, u32)>) -> Self {
        Self {
            air: XorLimbsAir { bus_index },
            pairs: Arc::new(Mutex::new(pairs)),
            xor_lookup_chip: XorLookupChip::<M>::new(bus_index),
        }
    }

    fn calc_xor(&self, a: u32, b: u32) -> u32 {
        a ^ b
    }

    pub fn request(&self, a: u32, b: u32) -> u32 {
        let mut pairs_locked = self.pairs.lock();
        pairs_locked.push((a, b));
        self.calc_xor(a, b)
    }
}
