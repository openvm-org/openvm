#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

use p3_field::Field;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct DummyHashChip {
    pub bus_index: usize,
    pub rate: usize,
    pub hash_width: usize,
}

impl DummyHashChip {
    pub fn request<F: Field>(curr_state: Vec<F>, _to_absorb: Vec<F>) -> Vec<F> {
        curr_state
    }

    pub fn get_width(&self) -> usize {
        2 * self.hash_width + self.rate
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }
}
