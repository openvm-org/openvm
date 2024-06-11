use std::sync::Arc;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

use crate::dummy_hash::DummyHashChip;
use p3_field::Field;
use parking_lot::Mutex;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct FlatHashAir {
    pub hash_chip_bus_index: usize,

    pub page_width: usize,
    pub page_height: usize,
    pub hash_width: usize,
    pub hash_rate: usize,
    pub digest_width: usize,

    // pub hash_chip: DummyHashAir,
    pub bus_index: usize,
}

pub struct PageController<F: Field> {
    pub air: FlatHashAir,
    pub hash_chip: Arc<Mutex<DummyHashChip<F>>>,
}

impl FlatHashAir {
    pub fn new(
        page_width: usize,
        page_height: usize,
        hash_width: usize,
        hash_rate: usize,
        digest_width: usize,
        hash_chip_bus_index: usize,
        bus_index: usize,
    ) -> Self {
        Self {
            hash_chip_bus_index,
            page_width,
            page_height,
            hash_width,
            hash_rate,
            digest_width,
            bus_index,
            // hash_chip: DummyHashAir::new(hashchip_bus_index, hash_width, hash_rate),
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn get_width(&self) -> usize {
        self.page_width + (self.page_width / self.hash_rate + 1) * self.hash_width
    }
}

impl<F: Field> PageController<F> {
    pub fn new(
        page_width: usize,
        page_height: usize,
        hash_width: usize,
        hash_rate: usize,
        digest_width: usize,
        hash_chip_bus_index: usize,
        bus_index: usize,
    ) -> Self {
        Self {
            air: FlatHashAir::new(
                page_width,
                page_height,
                hash_width,
                hash_rate,
                digest_width,
                hash_chip_bus_index,
                bus_index,
            ),
            hash_chip: Arc::new(Mutex::new(DummyHashChip::new(
                hash_chip_bus_index,
                hash_width,
                hash_rate,
            ))),
        }
    }
}
