use std::cmp::Ordering;

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{IsLessThanTupleBitsAuxCols, IsLessThanTupleBitsCols, IsLessThanTupleBitsIoCols},
    IsLessThanTupleBitsAir,
};
use crate::{
    is_equal::{columns::IsEqualAuxCols, IsEqualAir},
    is_less_than_bits::{columns::IsLessThanBitsAuxCols, IsLessThanBitsAir},
    sub_chip::LocalTraceInstructions,
};

impl IsLessThanTupleBitsAir {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        tuple_pairs: Vec<(Vec<u32>, Vec<u32>)>,
    ) -> RowMajorMatrix<F> {
        let num_cols: usize =
            IsLessThanTupleBitsCols::<F>::get_width(self.limb_bits().clone(), self.tuple_len());

        let mut rows: Vec<F> = vec![];

        // for each tuple pair, generate the trace row
        for (x, y) in tuple_pairs {
            let row: Vec<F> = self.generate_trace_row((x.clone(), y.clone())).flatten();
            rows.extend(row);
        }

        RowMajorMatrix::new(rows, num_cols)
    }
}

impl<F: PrimeField64> LocalTraceInstructions<F> for IsLessThanTupleBitsAir {
    type LocalInput = (Vec<u32>, Vec<u32>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y) = input;

        let mut less_than: Vec<F> = vec![];
        let mut less_than_aux: Vec<IsLessThanBitsAuxCols<F>> = vec![];
        let mut is_equal: Vec<F> = vec![];
        let mut is_equal_aux: Vec<IsEqualAuxCols<F>> = vec![];
        let mut less_than_cumulative: Vec<F> = vec![];

        for i in 0..x.len() {
            let is_less_than_bits_air = IsLessThanBitsAir::new(self.limb_bits()[i]);
            let curr_less_than_row =
                LocalTraceInstructions::generate_trace_row(&is_less_than_bits_air, (x[i], y[i]));
            less_than.push(curr_less_than_row.io.is_less_than);
            less_than_aux.push(curr_less_than_row.aux);

            let curr_is_equal_row = LocalTraceInstructions::generate_trace_row(
                &IsEqualAir {},
                (F::from_canonical_u32(x[i]), F::from_canonical_u32(y[i])),
            );
            is_equal.push(curr_is_equal_row.io.is_equal);
            is_equal_aux.push(curr_is_equal_row.aux);

            let less_than_here = match x[i].cmp(&y[i]) {
                Ordering::Less => F::one(),
                Ordering::Equal => {
                    if i > 0 {
                        less_than_cumulative[i - 1]
                    } else {
                        F::zero()
                    }
                }
                Ordering::Greater => F::zero(),
            };

            less_than_cumulative.push(less_than_here);
        }

        let tuple_less_than = less_than_cumulative[x.len() - 1];

        let io = IsLessThanTupleBitsIoCols {
            x: x.into_iter().map(F::from_canonical_u32).collect(),
            y: y.into_iter().map(F::from_canonical_u32).collect(),
            tuple_less_than,
        };
        let aux = IsLessThanTupleBitsAuxCols {
            less_than,
            less_than_aux,
            is_equal,
            is_equal_aux,
            less_than_cumulative,
        };

        IsLessThanTupleBitsCols { io, aux }
    }
}
