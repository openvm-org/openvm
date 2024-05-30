use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix; // Import the constant from columns.rs

use crate::sub_chip::LocalTraceInstructions;

use super::{columns::IsZeroCols, IsZeroAir};

impl IsZeroAir {
    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .x
            .iter()
            .map(|&x| {
                let iszerocols = self.generate_trace_row(x);
                vec![iszerocols.x, iszerocols.is_zero, iszerocols.inv]
            })
            .collect::<Vec<_>>();

        RowMajorMatrix::new(rows.concat(), IsZeroCols::<F>::get_width())
    }
}

impl<F: PrimeField32> LocalTraceInstructions<F> for IsZeroAir {
    type LocalInput = u32;

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let answer = if local_input == 0 { 1 } else { 0 };
        let inv = F::from_canonical_u32(local_input + answer).inverse();
        IsZeroCols::<F>::new(
            F::from_canonical_u32(local_input),
            F::from_canonical_u32(answer),
            inv,
        )
    }
}
