use std::sync::Arc;

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{IsLessThanTupleAuxColsMut, IsLessThanTupleCols, IsLessThanTupleColsMut},
    IsLessThanTupleAir, IsLessThanTupleChip,
};
use crate::{
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
    utils::{fill_slc_to_f, to_field_vec},
};

impl IsLessThanTupleChip {
    pub fn generate_trace<F: PrimeField>(
        &self,
        tuple_pairs: Vec<(Vec<u32>, Vec<u32>)>,
    ) -> RowMajorMatrix<F> {
        let width: usize = IsLessThanTupleCols::<F>::width(&self.air);

        let mut rows_concat: Vec<F> = vec![F::zero(); width * tuple_pairs.len()];

        // for each tuple pair, generate the trace row
        for (i, (x, y)) in tuple_pairs.iter().enumerate() {
            let mut cols = IsLessThanTupleColsMut::from_slice(
                &mut rows_concat[width * i..width * (i + 1)],
                &self.air,
            );

            self.air
                .generate_trace_row(x, y, &self.range_checker, &mut cols);
        }

        RowMajorMatrix::new(rows_concat, width)
    }
}

impl IsLessThanTupleAir {
    pub fn generate_trace_row<F: PrimeField>(
        &self,
        x: &[u32],
        y: &[u32],
        range_checker: &RangeCheckerGateChip,
        lt_cols: &mut IsLessThanTupleColsMut<F>,
    ) {
        fill_slc_to_f(lt_cols.io.x, x);
        fill_slc_to_f(lt_cols.io.y, y);
        *lt_cols.io.tuple_less_than = F::from_bool(x < y);

        self.generate_trace_row_aux(x, y, range_checker, &mut lt_cols.aux);
    }

    pub fn generate_trace_row_aux<F: PrimeField>(
        &self,
        x: &[u32],
        y: &[u32],
        range_checker: &RangeCheckerGateChip,
        lt_aux_cols: &mut IsLessThanTupleAuxColsMut<F>,
    ) {
        for i in 0..self.limb_bits.len() {
            lt_aux_cols.less_than[i] = F::from_bool(x[i] < y[i]);
            self.is_less_than_airs[i].generate_trace_row_aux(
                x[i],
                y[i],
                range_checker,
                &mut lt_aux_cols.less_than_aux[i],
            );
        }

        self.is_equal_vec_air.generate_trace_row_aux(
            to_field_vec(x).as_slice(),
            to_field_vec(y).as_slice(),
            &mut lt_aux_cols.is_equal_vec_aux,
        );

        *lt_aux_cols.is_equal_out = F::from_bool(x == y);

        let less_than_cumulative = &mut *lt_aux_cols.less_than_cumulative;

        // compute less_than_cumulative
        for i in 0..x.len() {
            let mut less_than_curr = if i > 0 {
                less_than_cumulative[i - 1]
            } else {
                F::zero()
            };

            if x[i] < y[i] && (i == 0 || lt_aux_cols.is_equal_vec_aux.prods[i - 1] == F::one()) {
                less_than_curr = F::one();
            }

            less_than_cumulative[i] = less_than_curr;
        }
    }
}

// TODO[jpw] stop using Arc<RangeCheckerGateChip> and use &RangeCheckerGateChip (requires not using this trait)
impl<F: PrimeField> LocalTraceInstructions<F> for IsLessThanTupleAir {
    type LocalInput = (Vec<u32>, Vec<u32>, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let width: usize = IsLessThanTupleCols::<F>::width(self);

        let mut row = vec![F::zero(); width];
        let mut lt_cols = IsLessThanTupleColsMut::<F>::from_slice(&mut row, self);

        self.generate_trace_row(
            input.0.as_slice(),
            input.1.as_slice(),
            &input.2,
            &mut lt_cols,
        );

        IsLessThanTupleCols::<F>::from_slice(&row, self)
    }
}
