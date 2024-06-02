use itertools::Itertools;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::IsEqualCols, IsEqualChip};

impl IsEqualChip {
    pub fn generate_trace<F: Field>(&self, x: Vec<F>, y: Vec<F>) -> RowMajorMatrix<F> {
        let rows = x
            .into_iter()
            .zip_eq(y)
            .flat_map(|(x, y)| {
                let is_equal_cols = self.generate_trace_row((x, y));
                [
                    is_equal_cols.io.x,
                    is_equal_cols.io.y,
                    is_equal_cols.io.is_equal,
                    is_equal_cols.inv,
                ]
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows, IsEqualCols::<F>::get_width())
    }
}

impl<F: Field> LocalTraceInstructions<F> for IsEqualChip {
    type LocalInput = (F, F);

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let is_equal = self.request(local_input.0, local_input.1);
        let inv = (local_input.0 - local_input.1 + F::from_bool(is_equal)).inverse();
        IsEqualCols::new(local_input.0, local_input.1, F::from_bool(is_equal), inv)
    }
}
