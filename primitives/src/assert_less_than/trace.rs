use std::borrow::BorrowMut;

use p3_field::{PrimeField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{AssertLessThanAuxCols, AssertLessThanCols},
    AssertLessThanAir, AssertLessThanChip,
};
use crate::var_range::VariableRangeCheckerChip;

impl<const AUX_LEN: usize> AssertLessThanChip<AUX_LEN> {
    pub fn generate_trace<F: PrimeField64>(&self, pairs: Vec<(u32, u32)>) -> RowMajorMatrix<F> {
        let width: usize = AssertLessThanCols::<F, AUX_LEN>::width();

        let mut rows_concat = vec![F::zero(); width * pairs.len()];
        for (i, (x, y)) in pairs.iter().enumerate() {
            let lt_cols: &mut AssertLessThanCols<F, AUX_LEN> =
                rows_concat[i * width..(i + 1) * width].borrow_mut();
            self.air
                .generate_trace_row(*x, *y, &self.range_checker, lt_cols);
        }

        RowMajorMatrix::new(rows_concat, width)
    }
}

impl<const AUX_LEN: usize> AssertLessThanAir<AUX_LEN> {
    pub fn generate_trace_row<F: PrimeField>(
        &self,
        x: u32,
        y: u32,
        range_checker: &VariableRangeCheckerChip,
        lt_cols: &mut AssertLessThanCols<F, AUX_LEN>,
    ) {
        lt_cols.io.x = F::from_canonical_u32(x);
        lt_cols.io.y = F::from_canonical_u32(y);

        self.generate_trace_row_aux(x, y, range_checker, &mut lt_cols.aux);
    }

    pub fn generate_trace_row_aux<F: PrimeField>(
        &self,
        x: u32,
        y: u32,
        range_checker: &VariableRangeCheckerChip,
        lt_aux_cols: &mut AssertLessThanAuxCols<F, AUX_LEN>,
    ) {
        // if x >= y then no valid trace exists
        assert!(x < y);

        // obtain the lower_bits
        let check_less_than = y - x - 1;
        let lower_u32 = check_less_than & ((1 << self.max_bits) - 1);
        let num_limbs = AssertLessThanAuxCols::<F, AUX_LEN>::width();
        // decompose lower_bits into limbs and range check
        for i in 0..num_limbs {
            let bits =
                (lower_u32 >> (i * self.bus.range_max_bits)) & ((1 << self.bus.range_max_bits) - 1);
            lt_aux_cols.lower_decomp[i] = F::from_canonical_u32(bits);

            if i == num_limbs - 1 && self.max_bits % self.bus.range_max_bits != 0 {
                let last_limb_max_bits = self.max_bits % self.bus.range_max_bits;
                range_checker.add_count(bits, last_limb_max_bits);
            } else {
                range_checker.add_count(bits, self.bus.range_max_bits);
            }
        }
    }
}
