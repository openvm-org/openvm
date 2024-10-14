use std::sync::Arc;

use air::XorRequesterAir;

use crate::xor::{bits::XorBitsChip, bus::XorBus};

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
    pub fn new(bus: XorBus, requests: Vec<(u32, u32)>, xor_chip: Arc<XorBitsChip<N>>) -> Self {
        Self {
            air: XorRequesterAir { bus },
            requests,
            xor_chip,
        }
    }

    /// The xor bus this chip interacts with
    pub fn bus(&self) -> XorBus {
        self.air.bus
    }

    pub fn add_request(&mut self, a: u32, b: u32) {
        self.requests.push((a, b));
    }
}
