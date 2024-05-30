use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::IsEqualVecCols, IsEqualVecChip};

impl IsEqualVecChip {
    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let width: usize = self.get_width();
        let height: usize = self.x.len();
        assert!(height.is_power_of_two());
        // let mut rows = Vec::with_capacity(height);

        let rows: Vec<Vec<F>> = self
            .x
            .iter()
            .zip(self.y.iter())
            .map(|(x_row, y_row)| {
                let row = self.generate_trace_row((x_row.clone(), y_row.clone()));
                row.to_vec()
            })
            .collect();

        RowMajorMatrix::new(rows.concat(), width)
    }
}

impl<F: PrimeField32> LocalTraceInstructions<F> for IsEqualVecChip {
    type LocalInput = (Vec<u32>, Vec<u32>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> IsEqualVecCols<F> {
        let (x_row, y_row) = input;
        let vec_len = self.vec_len();
        let mut transition_index = 0;
        while transition_index < vec_len && x_row[transition_index] == y_row[transition_index] {
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

        IsEqualVecCols::from_slice(&row, vec_len)
    }
}
