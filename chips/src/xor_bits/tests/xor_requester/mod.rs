use std::sync::Arc;

use air::XorRequesterAir;

use crate::xor_bits::XorBitsChip;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Debug)]
pub struct XorRequesterChip<const N: usize> {
    pub air: XorRequesterAir,
    pub requests: Vec<(u32, u32)>,

    xor_chip: Arc<XorBitsChip<N>>,
}

impl<const N: usize> XorRequesterChip<N> {
    pub fn new(bus_index: usize, requests: Vec<(u32, u32)>, xor_chip: Arc<XorBitsChip<N>>) -> Self {
        Self {
            air: XorRequesterAir { bus_index },
            requests,
            xor_chip,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.air.bus_index
    }

    pub fn add_request(&mut self, a: u32, b: u32) {
        self.requests.push((a, b));
    }
}
