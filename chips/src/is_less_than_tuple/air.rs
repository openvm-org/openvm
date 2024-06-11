use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_equal::{
        columns::{IsEqualAuxCols, IsEqualCols, IsEqualIOCols},
        IsEqualAir,
    },
    is_less_than::columns::{IsLessThanAuxCols, IsLessThanCols, IsLessThanIOCols},
    sub_chip::{AirConfig, SubAir},
};

use super::{
    columns::{IsLessThanTupleAuxCols, IsLessThanTupleCols, IsLessThanTupleIOCols},
    IsLessThanTupleAir,
};

impl AirConfig for IsLessThanTupleAir {
    type Cols<T> = IsLessThanTupleCols<T>;
}

impl<F: Field> BaseAir<F> for IsLessThanTupleAir {
    fn width(&self) -> usize {
        IsLessThanTupleCols::<F>::get_width(
            self.limb_bits().clone(),
            self.decomp(),
            self.tuple_len(),
        )
    }
}

impl<AB: AirBuilder> Air<AB> for IsLessThanTupleAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsLessThanTupleCols::<AB::Var>::from_slice(
            local,
            self.limb_bits().clone(),
            self.decomp(),
            self.tuple_len(),
        );

        SubAir::eval(self, builder, local_cols.io, local_cols.aux);
    }
}

// sub-chip with constraints to check whether one tuple is less than the another
impl<AB: AirBuilder> SubAir<AB> for IsLessThanTupleAir {
    type IoView = IsLessThanTupleIOCols<AB::Var>;
    type AuxView = IsLessThanTupleAuxCols<AB::Var>;

    // constrain that x < y lexicographically
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let x = io.x.clone();
        let y = io.y.clone();

        // here we constrain that less_than[i] indicates whether x[i] < y[i] using the IsLessThan subchip for each i
        for i in 0..x.len() {
            let x_val = x[i];
            let y_val = y[i];

            let is_less_than_cols = IsLessThanCols {
                io: IsLessThanIOCols {
                    x: x_val,
                    y: y_val,
                    less_than: aux.less_than[i],
                },
                aux: IsLessThanAuxCols {
                    lower: aux.less_than_aux[i].lower,
                    lower_decomp: aux.less_than_aux[i].lower_decomp.clone(),
                },
            };

            SubAir::eval(
                &self.is_less_than_airs[i].clone(),
                builder,
                is_less_than_cols.io,
                is_less_than_cols.aux,
            );
        }

        // here, we constrain that is_equal is the indicator for whether diff == 0, i.e. x[i] = y[i]
        for i in 0..x.len() {
            let is_equal = aux.is_equal[i];
            let inv = aux.is_equal_aux[i].inv;

            let is_equal_cols = IsEqualCols {
                io: IsEqualIOCols {
                    x: x[i],
                    y: y[i],
                    is_equal,
                },
                aux: IsEqualAuxCols { inv },
            };

            SubAir::eval(&IsEqualAir, builder, is_equal_cols.io, is_equal_cols.aux);
        }

        // here, we constrain that is_equal_cumulative and less_than_cumulative are the correct values
        let is_equal_cumulative = aux.is_equal_cumulative.clone();
        let less_than_cumulative = aux.less_than_cumulative.clone();

        builder.assert_eq(is_equal_cumulative[0], aux.is_equal[0]);
        builder.assert_eq(less_than_cumulative[0], aux.less_than[0]);

        for i in 1..x.len() {
            // this constrains that is_equal_cumulative[i] indicates whether the first i elements of x and y are equal
            builder.assert_eq(
                is_equal_cumulative[i],
                is_equal_cumulative[i - 1] * aux.is_equal[i],
            );
            // this constrains that less_than_cumulative[i] indicates whether the first i elements of x are less than
            // the first i elements of y, lexicographically
            // note that less_than_cumulative[i - 1] and is_equal_cumulative[i - 1] are never both 1
            builder.assert_eq(
                less_than_cumulative[i],
                less_than_cumulative[i - 1] + aux.less_than[i] * is_equal_cumulative[i - 1],
            );
        }

        // constrain that the tuple_less_than does indicate whether x < y, lexicographically
        builder.assert_eq(io.tuple_less_than, less_than_cumulative[x.len() - 1]);
    }
}
