use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix; // Import the constant from columns.rs

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::IsZeroCols, IsZeroChip};

impl IsZeroChip {
    pub fn generate_trace<F: Field>(&self, x: Vec<F>) -> RowMajorMatrix<F> {
        let rows = x
            .iter()
            .flat_map(|&x| {
                let is_zero_cols = self.generate_trace_row(x);
                [is_zero_cols.io.x, is_zero_cols.io.is_zero, is_zero_cols.inv]
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows, IsZeroCols::<F>::get_width())
    }
}

impl<F: Field> LocalTraceInstructions<F> for IsZeroChip {
    type LocalInput = F;

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let is_zero = IsZeroChip::request(local_input);
        let inv = if is_zero {
            F::zero()
        } else {
            local_input.inverse()
        };
        IsZeroCols::<F>::new(local_input, F::from_bool(is_zero), inv)
    }
}
