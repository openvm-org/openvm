use std::sync::Arc;

use crate::xor_bits::XorBitsChip;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct XorRequesterChip<const N: usize> {
    /// The index for the Xor Chip bus.
    bus_index: usize,
    pub requests: Vec<(u32, u32)>,

    xor_chip: Arc<XorBitsChip<N>>,
}

impl<const N: usize> XorRequesterChip<N> {
    pub fn new(bus_index: usize, requests: Vec<(u32, u32)>, xor_chip: Arc<XorBitsChip<N>>) -> Self {
        Self {
            bus_index,
            requests,
            xor_chip,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn add_request(&mut self, a: u32, b: u32) {
        self.requests.push((a, b));
    }
}
