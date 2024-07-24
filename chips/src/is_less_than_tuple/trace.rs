use std::sync::Arc;

use p3_field::{PrimeField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    is_equal_vec::columns::IsEqualVecAuxCols,
    is_less_than::{columns::IsLessThanAuxCols, IsLessThanChip},
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};

use super::{
    columns::{IsLessThanTupleAuxCols, IsLessThanTupleCols, IsLessThanTupleIoCols},
    IsLessThanTupleAir, IsLessThanTupleChip,
};

impl IsLessThanTupleChip {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        tuple_pairs: Vec<(Vec<u32>, Vec<u32>)>,
    ) -> RowMajorMatrix<F> {
        let num_cols: usize = IsLessThanTupleCols::<F>::width(&self.air);

        let mut rows: Vec<F> = vec![];

        // for each tuple pair, generate the trace row
        for (x, y) in tuple_pairs {
            let row: Vec<F> = self
                .air
                .generate_trace_row((x.clone(), y.clone(), self.range_checker.clone()))
                .flatten();
            rows.extend(row);
        }

        RowMajorMatrix::new(rows, num_cols)
    }
}

impl<F: PrimeField> LocalTraceInstructions<F> for IsLessThanTupleAir {
    type LocalInput = (Vec<u32>, Vec<u32>, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;

        let mut less_than: Vec<F> = vec![];
        let mut lower_vec: Vec<F> = vec![];
        let mut lower_decomp_vec: Vec<Vec<F>> = vec![];

        let mut valid = true;
        let mut tuple_less_than = F::zero();

        // use subchip to generate relevant columns
        for (i, &limb_bits) in self.limb_bits.iter().enumerate() {
            let is_less_than_chip = IsLessThanChip::new(
                self.bus_index,
                limb_bits,
                self.decomp,
                range_checker.clone(),
            );

            let curr_less_than_row = LocalTraceInstructions::generate_trace_row(
                &is_less_than_chip.air,
                (x[i], y[i], range_checker.clone()),
            )
            .flatten();
            less_than.push(curr_less_than_row[2]);
            lower_vec.push(curr_less_than_row[3]);
            lower_decomp_vec.push(curr_less_than_row[4..].to_vec());
        }

        // compute prods and invs
        let mut transition_index = 0;
        while transition_index < x.len() && x[transition_index] == y[transition_index] {
            transition_index += 1;
        }

        let prods = std::iter::repeat(F::one())
            .take(transition_index)
            .chain(std::iter::repeat(F::zero()).take(x.len() - transition_index))
            .collect::<Vec<F>>();

        let mut invs = std::iter::repeat(F::zero())
            .take(x.len())
            .collect::<Vec<F>>();

        if transition_index != x.len() {
            invs[transition_index] = (F::from_canonical_u32(x[transition_index])
                - F::from_canonical_u32(y[transition_index]))
            .inverse();
        }

        let mut less_than_cumulative: Vec<F> = vec![];

        // compute less_than_cumulative
        for i in 0..x.len() {
            let mut less_than_curr = if i > 0 {
                less_than_cumulative[i - 1]
            } else {
                F::zero()
            };

            if x[i] < y[i] && (i == 0 || prods[i - 1] == F::one()) {
                less_than_curr = F::one();
            }

            if x[i] < y[i] && valid {
                tuple_less_than = F::one();
            } else if x[i] > y[i] && valid {
                valid = false;
            }

            less_than_cumulative.push(less_than_curr);
        }

        // compute less_than_aux and is_equal_vec_aux
        let mut less_than_aux: Vec<IsLessThanAuxCols<F>> = vec![];
        for i in 0..x.len() {
            let less_than_col = IsLessThanAuxCols {
                lower: lower_vec[i],
                lower_decomp: lower_decomp_vec[i].clone(),
            };
            less_than_aux.push(less_than_col);
        }

        let is_equal_vec_aux = IsEqualVecAuxCols { prods, invs };

        let io = IsLessThanTupleIoCols {
            x: x.into_iter().map(F::from_canonical_u32).collect(),
            y: y.into_iter().map(F::from_canonical_u32).collect(),
            tuple_less_than,
        };
        let aux = IsLessThanTupleAuxCols {
            less_than,
            less_than_aux,
            is_equal_vec_aux,
            less_than_cumulative,
        };

        IsLessThanTupleCols { io, aux }
    }
}
