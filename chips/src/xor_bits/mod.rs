use parking_lot::Mutex;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

#[derive(Default)]
/// A chip that computes the xor of two numbers of at most N bits each
pub struct XorBitsChip<const N: usize> {
    bus_index: usize,

    /// List of all requests sent to the chip
    pairs: Mutex<Vec<(u32, u32)>>,
}

impl<const N: usize> XorBitsChip<N> {
    pub fn new(bus_index: usize, pairs: Vec<(u32, u32)>) -> Self {
        Self {
            bus_index,
            pairs: Mutex::new(pairs),
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
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
