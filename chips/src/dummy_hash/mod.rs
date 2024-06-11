#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

use p3_field::Field;

#[derive(Default)]
/// The AIR for the dummy hash chip
pub struct DummyHashAir {
    pub bus_index: usize,
    pub rate: usize,
    pub hash_width: usize,
}

#[derive(Default)]
pub struct DummyHashChip<F: Field> {
    pub air: DummyHashAir,
    pub hash_in_states: Vec<Vec<F>>,
    pub hash_slices: Vec<Vec<F>>,
    pub hash_out_states: Vec<Vec<F>>,
}

impl DummyHashAir {
    pub fn new(bus_index: usize, hash_width: usize, rate: usize) -> Self {
        Self {
            bus_index,
            rate,
            hash_width,
        }
    }

    pub fn hash<F: Field>(curr_state: Vec<F>, to_absorb: Vec<F>) -> Vec<F> {
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
        2 * self.hash_width + self.rate + 1
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }
}

impl<F: Field> DummyHashChip<F> {
    pub fn new(bus_index: usize, hash_width: usize, rate: usize) -> Self {
        Self {
            air: DummyHashAir::new(bus_index, hash_width, rate),
            hash_in_states: vec![],
            hash_slices: vec![],
            hash_out_states: vec![],
        }
    }

    pub fn request(&mut self, curr_state: Vec<F>, to_absorb: Vec<F>) -> Vec<F> {
        let new_state = DummyHashAir::hash(curr_state.clone(), to_absorb.clone());

        self.hash_in_states.push(curr_state);
        self.hash_slices.push(to_absorb);
        self.hash_out_states.push(new_state.clone());

        new_state
    }
}
