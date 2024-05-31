use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::IsEqualCols, IsEqualChip};

impl IsEqualChip {
    pub fn generate_trace_rows<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .x
            .iter()
            .enumerate()
            .map(|(i, _x)| {
                let is_equal_cols = self.generate_trace_row((self.x[i], self.y[i]));
                vec![
                    is_equal_cols.io.x,
                    is_equal_cols.io.y,
                    is_equal_cols.io.is_equal,
                    is_equal_cols.inv,
                ]
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows.concat(), IsEqualCols::<F>::get_width())
    }
}

impl<F: PrimeField32> LocalTraceInstructions<F> for IsEqualChip {
    type LocalInput = (u32, u32);

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let is_equal = self.is_equal(local_input.0, local_input.1);
        let inv = F::from_canonical_u32(local_input.0 - local_input.1 + is_equal).inverse();
        IsEqualCols::<F>::new(
            F::from_canonical_u32(local_input.0),
            F::from_canonical_u32(local_input.1),
            F::from_canonical_u32(is_equal),
            inv,
        )
    }
}
