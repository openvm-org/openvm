pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

use self::columns::Poseidon2Cols;
use p3_matrix::dense::RowMajorMatrix;

pub struct Poseidon2Air<const WIDTH: usize, T: Clone> {
    pub rounds_f: usize,
    pub external_constants: Vec<[T; WIDTH]>,
    pub rounds_p: usize,
    pub internal_constants: Vec<T>,
    pub BUS_INDEX: usize,
    pub trace: Option<RowMajorMatrix<T>>,
}

impl<const WIDTH: usize, T: Clone> Poseidon2Air<WIDTH, T> {
    pub fn new(
        external_constants: Vec<[T; WIDTH]>,
        internal_constants: Vec<T>,
        bus_index: usize,
    ) -> Self {
        Self {
            rounds_f: external_constants.len(),
            external_constants,
            rounds_p: internal_constants.len(),
            internal_constants,
            BUS_INDEX: bus_index,
            trace: None,
        }
    }
    pub fn get_width(&self) -> usize {
        Poseidon2Cols::<WIDTH, T>::get_width(self)
    }
}
