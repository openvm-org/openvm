use std::sync::Arc;

use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{IsLessThanAuxColsMut, IsLessThanCols, IsLessThanColsMut},
    IsLessThanAir, IsLessThanChip,
};
use crate::{sub_chip::LocalTraceInstructions, var_range::VariableRangeCheckerChip};

impl IsLessThanChip {
    pub fn generate_trace<F: Field>(&self, pairs: Vec<(u32, u32)>) -> RowMajorMatrix<F> {
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
    pub fn generate_trace_row<F: Field>(
        &self,
        x: u32,
        y: u32,
        range_checker: &VariableRangeCheckerChip,
        lt_cols: &mut IsLessThanColsMut<F>,
    ) {
        let less_than = if x < y { 1 } else { 0 };

        *lt_cols.io.x = F::from_canonical_u32(x);
        *lt_cols.io.y = F::from_canonical_u32(y);
        *lt_cols.io.less_than = F::from_canonical_u32(less_than);

        self.generate_trace_row_aux(x, y, range_checker, &mut lt_cols.aux);
    }

    pub fn generate_trace_row_aux<F: Field>(
        &self,
        x: u32,
        y: u32,
        range_checker: &VariableRangeCheckerChip,
        lt_aux_cols: &mut IsLessThanAuxColsMut<F>,
    ) {
        // obtain the lower_bits
        let check_less_than = (1 << self.max_bits) + y - x - 1;
        let lower_u32 = check_less_than & ((1 << self.max_bits) - 1);

        // decompose lower_bits into limbs and range check
        let mut lower_decomp: Vec<F> = Vec::with_capacity(self.num_limbs);
        let mask = (1 << self.range_max_bits()) - 1;
        let mut bits_remaining = self.max_bits;
        for i in 0..self.num_limbs {
            let limb = (lower_u32 >> (i * self.range_max_bits())) & mask;

            lower_decomp.push(F::from_canonical_u32(limb));
            range_checker.add_count(limb, bits_remaining.min(self.range_max_bits()));
            bits_remaining = bits_remaining.saturating_sub(self.range_max_bits());
        }

        lt_aux_cols
            .lower_decomp
            .copy_from_slice(lower_decomp.as_slice());
    }
}

// TODO[jpw] stop using Arc<VariableRangeCheckerChip> and use &VariableRangeCheckerChip (requires not using this trait)
impl<F: Field> LocalTraceInstructions<F> for IsLessThanAir {
    type LocalInput = (u32, u32, Arc<VariableRangeCheckerChip>);

    fn generate_trace_row(
        &self,
        (x, y, range_checker): (u32, u32, Arc<VariableRangeCheckerChip>),
    ) -> Self::Cols<F> {
        let width: usize = IsLessThanCols::<F>::width(self);

        let mut row = vec![F::zero(); width];
        let mut lt_cols = IsLessThanColsMut::<F>::from_slice(&mut row);

        self.generate_trace_row(x, y, &range_checker, &mut lt_cols);

        IsLessThanCols::<F>::from_slice(&row)
    }
}
