use std::sync::Arc;

use p3_field::{PrimeField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{IsLessThanAuxColsMut, IsLessThanCols, IsLessThanColsMut},
    IsLessThanAir, IsLessThanChip,
};
use crate::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};

impl IsLessThanChip {
    pub fn generate_trace<F: PrimeField64>(&self, pairs: Vec<(u32, u32)>) -> RowMajorMatrix<F> {
        let width: usize = IsLessThanCols::<F>::width(&self.air);

        let mut rows_concat = vec![F::zero(); width * pairs.len()];
        for (i, (x, y)) in pairs.iter().enumerate() {
            let mut lt_cols =
                IsLessThanColsMut::<F>::from_slice(&mut rows_concat[i * width..(i + 1) * width]);

            self.air
                .generate_trace_row(*x, *y, &self.range_checker, &mut lt_cols);
        }

        RowMajorMatrix::new(rows_concat, width)
    }
}

impl IsLessThanAir {
    pub fn generate_trace_row<F: PrimeField>(
        &self,
        x: u32,
        y: u32,
        range_checker: &RangeCheckerGateChip,
        lt_cols: &mut IsLessThanColsMut<F>,
    ) {
        let less_than = if x < y { 1 } else { 0 };

        *lt_cols.io.x = F::from_canonical_u32(x);
        *lt_cols.io.y = F::from_canonical_u32(y);
        *lt_cols.io.less_than = F::from_canonical_u32(less_than);

        self.generate_trace_row_aux(x, y, range_checker, &mut lt_cols.aux);
    }

    pub fn generate_trace_row_aux<F: PrimeField>(
        &self,
        x: u32,
        y: u32,
        range_checker: &RangeCheckerGateChip,
        lt_aux_cols: &mut IsLessThanAuxColsMut<F>,
    ) {
        // obtain the lower_bits
        let check_less_than = (1 << self.max_bits) + y - x - 1;
        let lower_u32 = check_less_than & ((1 << self.max_bits) - 1);

        // decompose lower_bits into limbs and range check
        let mut lower_decomp: Vec<F> =
            Vec::with_capacity(self.num_limbs + (self.max_bits % self.decomp != 0) as usize);
        for i in 0..self.num_limbs {
            let bits = (lower_u32 >> (i * self.decomp)) & ((1 << self.decomp) - 1);

            lower_decomp.push(F::from_canonical_u32(bits));
            range_checker.add_count(bits);

            if i == self.num_limbs - 1 && self.max_bits % self.decomp != 0 {
                let last_limb_shift = (self.decomp - (self.max_bits % self.decomp)) % self.decomp;
                let last_limb_shifted = bits << last_limb_shift;

                lower_decomp.push(F::from_canonical_u32(last_limb_shifted));
                range_checker.add_count(last_limb_shifted);
            }
        }

        lt_aux_cols
            .lower_decomp
            .clone_from_slice(lower_decomp.as_slice());
    }
}

// TODO[jpw] stop using Arc<RangeCheckerGateChip> and use &RangeCheckerGateChip (requires not using this trait)
impl<F: PrimeField> LocalTraceInstructions<F> for IsLessThanAir {
    type LocalInput = (u32, u32, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: (u32, u32, Arc<RangeCheckerGateChip>)) -> Self::Cols<F> {
        let width: usize = IsLessThanCols::<F>::width(self);

        let mut row = vec![F::zero(); width];
        let mut lt_cols = IsLessThanColsMut::<F>::from_slice(&mut row);

        self.generate_trace_row(input.0, input.1, &input.2, &mut lt_cols);

        IsLessThanCols::<F>::from_slice(&row)
    }
}
