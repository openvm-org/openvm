pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct XorChip<const N: usize> {
    bus_index: usize,

    pub pairs: Vec<(u32, u32)>,
}

impl<const N: usize> XorChip<N> {
    pub fn new(bus_index: usize, pairs: Vec<(u32, u32)>) -> Self {
        Self { bus_index, pairs }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn get_width(&self) -> usize {
        3 * N + 3
    }

    pub fn calc_xor(&self, a: u32, b: u32) -> u32 {
        a ^ b
    }

    pub fn request(&mut self, a: u32, b: u32) -> u32 {
        self.pairs.push((a, b));
        self.calc_xor(a, b)
    }
}
