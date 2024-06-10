#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

use crate::dummy_hash::DummyHashChip;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct FlatHashChip<const N: usize, const R: usize> {
    pub hashchip_bus_index: usize,

    pub page_width: usize,
    pub page_height: usize,
    pub hash_width: usize,
    pub hash_rate: usize,
    pub digest_width: usize,

    hashchip: DummyHashChip,
    bus_index: usize,
}

impl<const N: usize, const R: usize> FlatHashChip<N, R> {
    pub fn new(
        page_width: usize,
        page_height: usize,
        hash_width: usize,
        hash_rate: usize,
        digest_width: usize,
        hashchip_bus_index: usize,
        bus_index: usize,
    ) -> Self {
        Self {
            hashchip_bus_index,
            page_width,
            page_height,
            hash_width,
            hash_rate,
            digest_width,
            bus_index,
            hashchip: DummyHashChip {
                bus_index: hashchip_bus_index,
                hash_width,
                rate: hash_rate,
            },
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn get_width(&self) -> usize {
        self.page_width + (self.page_width / self.hash_rate + 1) * self.hash_width
    }
}
