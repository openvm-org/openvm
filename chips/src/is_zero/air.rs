use std::borrow::Borrow;

use super::columns::{IsZeroCols, NUM_COLS};
use super::IsZeroChip;
use crate::sub_chip::SubAir;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;

impl<F: Field> BaseAir<F> for IsZeroChip {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for IsZeroChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let is_zero_cols = IsZeroCols::from_slice(local);

        SubAir::<AB>::eval(self, builder, is_zero_cols);
    }
}

impl<AB: AirBuilder> SubAir<AB> for IsZeroChip {
    fn eval(&self, builder: &mut AB, is_zero_cols: Self::Cols<AB::Var>) {
        builder.assert_eq(is_zero_cols.x * is_zero_cols.is_zero, AB::F::zero());
        builder.assert_eq(
            is_zero_cols.is_zero + is_zero_cols.x * is_zero_cols.inv,
            AB::F::one(),
        );
    }
}
