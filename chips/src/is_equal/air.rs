use std::borrow::Borrow;

use super::columns::{IsEqualCols, NUM_COLS};
use super::IsEqualChip;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
// use p3_field::Field;
use crate::sub_chip::SubAir;
use p3_matrix::Matrix;

impl<F> BaseAir<F> for IsEqualChip {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for IsEqualChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let is_equal_cols = IsEqualCols::from_slice(local);

        SubAir::<AB>::eval(self, builder, is_equal_cols);
    }
}

impl<AB: AirBuilder> SubAir<AB> for IsEqualChip {
    fn eval(&self, builder: &mut AB, is_equal_cols: Self::Cols<AB::Var>) {
        builder.assert_eq(
            (is_equal_cols.x - is_equal_cols.y + is_equal_cols.is_equal) * is_equal_cols.inv,
            AB::F::one(),
        );
        builder.assert_eq(
            is_equal_cols.is_equal * is_equal_cols.is_equal,
            is_equal_cols.is_equal,
        );
        builder.assert_eq(
            (is_equal_cols.x - is_equal_cols.y) * is_equal_cols.is_equal,
            AB::F::zero(),
        );
    }
}
