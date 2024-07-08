use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::Poseidon2Chip;

impl<const WIDTH: usize, F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    /// Generates trace for poseidon2chip from cached row structs.
    /// TODO: add blank rows to pad length
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            self.rows.iter().flat_map(|row| row.flatten()).collect(),
            self.width(),
        )
    }
}
