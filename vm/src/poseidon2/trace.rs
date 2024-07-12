use super::columns::Poseidon2ChipCols;
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::Poseidon2Chip;

impl<const WIDTH: usize, F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    /// Generates trace for poseidon2chip from cached row structs.
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        let row_len = self.rows.len();
        let correct_len = row_len.next_power_of_two();
        let blank_row = Poseidon2ChipCols::<WIDTH, F>::blank_row(&self.air).flatten();
        let diff = correct_len - row_len;
        RowMajorMatrix::new(
            self.rows
                .iter()
                .flat_map(|row| row.flatten())
                .chain(std::iter::repeat(blank_row.clone()).take(diff).flatten())
                .collect(),
            self.width(),
        )
    }
}
