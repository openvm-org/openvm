use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix; // Import the constant from columns.rs

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::IsZeroCols, IsZeroChip};

impl<F: Field> IsZeroChip<F> {
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        let rows = self
            .x
            .iter()
            .map(|&x| {
                let is_zero_cols = self.generate_trace_row(x);
                vec![is_zero_cols.io.x, is_zero_cols.io.is_zero, is_zero_cols.inv]
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows.concat(), IsZeroCols::<F>::get_width())
    }
}

impl<F: Field> LocalTraceInstructions<F> for IsZeroChip<F> {
    type LocalInput = F;

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let is_zero = self.request(local_input);
        let inv = if is_zero {
            F::zero()
        } else {
            local_input.inverse()
        };
        IsZeroCols::<F>::new(local_input, F::from_bool(is_zero), inv)
    }
}
