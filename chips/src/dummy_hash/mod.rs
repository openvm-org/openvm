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
    pub fn request<F: Field>(&self, curr_state: Vec<F>, to_absorb: Vec<F>) -> Vec<F> {
        let mut new_state = curr_state.clone();

        for (new, b) in new_state
            .iter_mut()
            .take(to_absorb.len())
            .zip(to_absorb.iter())
        {
            *new += *b;
        }

        new_state
    }

    pub fn get_width(&self) -> usize {
        2 * self.hash_width + self.rate
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }
}
