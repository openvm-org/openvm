use std::sync::Arc;

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    is_equal::columns::IsEqualAuxCols,
    is_less_than::{columns::IsLessThanAuxCols, IsLessThanChip},
    range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};

use super::{
    columns::{IsLessThanTupleAuxCols, IsLessThanTupleCols, IsLessThanTupleIOCols},
    IsLessThanTupleAir, IsLessThanTupleChip,
};

impl IsLessThanTupleChip {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        tuple_pairs: Vec<(Vec<u32>, Vec<u32>)>,
    ) -> RowMajorMatrix<F> {
        let num_cols: usize = IsLessThanTupleCols::<F>::get_width(
            self.air.limb_bits().clone(),
            self.air.decomp(),
            self.air.tuple_len(),
        );

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

impl<F: PrimeField64> LocalTraceInstructions<F> for IsLessThanTupleAir {
    type LocalInput = (Vec<u32>, Vec<u32>, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;

        let mut less_than: Vec<F> = vec![];
        let mut lower_vec: Vec<F> = vec![];
        let mut lower_decomp_vec: Vec<Vec<F>> = vec![];

        let mut valid = true;
        let mut tuple_less_than = F::zero();

        // use subchip to generate relevant columns
        for i in 0..x.len() {
            let is_less_than_chip = IsLessThanChip::new(
                self.bus_index(),
                self.range_max(),
                self.limb_bits()[i],
                self.decomp(),
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

        // compute is_equal_cumulative
        let mut transition_index = 0;
        while transition_index < x.len() && x[transition_index] == y[transition_index] {
            transition_index += 1;
        }

        let is_equal_cumulative = std::iter::repeat(F::one())
            .take(transition_index)
            .chain(std::iter::repeat(F::zero()).take(x.len() - transition_index))
            .collect::<Vec<F>>();

        let mut less_than_cumulative: Vec<F> = vec![];

        // compute less_than_cumulative
        for i in 0..x.len() {
            let mut less_than_curr = if i > 0 {
                less_than_cumulative[i - 1]
            } else {
                F::zero()
            };

            if x[i] < y[i] && (i == 0 || is_equal_cumulative[i - 1] == F::one()) {
                less_than_curr = F::one();
            }

            if x[i] < y[i] && valid {
                tuple_less_than = F::one();
            } else if x[i] > y[i] && valid {
                valid = false;
            }

            less_than_cumulative.push(less_than_curr);
        }

        // contains indicator whether difference is zero
        let mut is_equal: Vec<F> = vec![];
        // contains y such that y * (i + x) = 1
        let mut inverses: Vec<F> = vec![];

        // we compute the indicators, which only matter if the row is not the last
        for (i, &val) in x.iter().enumerate() {
            let next_val = y[i];

            // the difference between the two limbs
            let curr_diff = F::from_canonical_u32(val) - F::from_canonical_u32(next_val);

            // compute the equal indicator and inverses
            if next_val == val {
                is_equal.push(F::one());
                inverses.push((curr_diff + F::one()).inverse());
            } else {
                is_equal.push(F::zero());
                inverses.push(curr_diff.inverse());
            }
        }

        // compute less_than_aux and is_equal_aux
        let mut less_than_aux: Vec<IsLessThanAuxCols<F>> = vec![];
        for i in 0..x.len() {
            let less_than_col = IsLessThanAuxCols {
                lower: lower_vec[i],
                lower_decomp: lower_decomp_vec[i].clone(),
            };
            less_than_aux.push(less_than_col);
        }

        let mut is_equal_aux: Vec<IsEqualAuxCols<F>> = vec![];
        for inverse in &inverses {
            let is_equal_col = IsEqualAuxCols { inv: *inverse };
            is_equal_aux.push(is_equal_col);
        }

        let io = IsLessThanTupleIOCols {
            x: x.into_iter().map(F::from_canonical_u32).collect(),
            y: y.into_iter().map(F::from_canonical_u32).collect(),
            tuple_less_than,
        };
        let aux = IsLessThanTupleAuxCols {
            less_than,
            less_than_aux,
            is_equal,
            is_equal_aux,
            is_equal_cumulative,
            less_than_cumulative,
        };

        IsLessThanTupleCols { io, aux }
    }
}
