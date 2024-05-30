use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::IsEqualVecChip;

impl IsEqualVecChip {
    pub fn generate_trace_rows<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let width: usize = self.get_width();
        let vec_len: usize = self.vec_len();
        let height: usize = self.x.len();
        assert!(height.is_power_of_two());
        // let mut rows = Vec::with_capacity(height);

        // TODO make sexy
        let rows: Vec<Vec<F>> = self
            .x
            .iter()
            .zip(self.y.iter())
            .map(|(x_row, y_row)| {
                let mut transition_index = 0;
                while transition_index < vec_len
                    && x_row[transition_index] == y_row[transition_index]
                {
                    transition_index += 1;
                }

                let mut row = x_row
                    .iter()
                    .chain(y_row.iter())
                    .map(|&val| F::from_canonical_u32(val))
                    .chain(std::iter::repeat(F::one()).take(transition_index))
                    .chain(std::iter::repeat(F::zero()).take(2 * vec_len - transition_index))
                    .collect::<Vec<F>>();

                if transition_index != vec_len {
                    row[3 * vec_len + transition_index] =
                        (row[transition_index] - row[transition_index + vec_len]).inverse();
                }

                row
            })
            .collect();

        RowMajorMatrix::new(rows.concat(), width)
    }
}
