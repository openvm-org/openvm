pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

use crate::mbit_xor::MBitXorChip;

#[derive(Default)]
pub struct NBitXorChip<const N: usize, const M: usize> {
    bus_index: usize,

    pub pairs: Vec<(u32, u32)>,
    pub mbit_xor_chip: MBitXorChip<M>,
}

impl<const N: usize, const M: usize> NBitXorChip<N, M> {
    pub fn new(bus_index: usize, pairs: Vec<(u32, u32)>) -> Self {
        Self {
            bus_index,
            pairs,
            mbit_xor_chip: MBitXorChip::<M>::new(bus_index),
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    fn calc_xor(&self, a: u32, b: u32) -> u32 {
        a ^ b
    }

    pub fn request(&mut self, a: u32, b: u32) -> u32 {
        self.pairs.push((a, b));
        self.calc_xor(a, b)
    }
}
