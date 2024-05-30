use std::borrow::Borrow;

use super::columns::{IsZeroCols, NUM_COLS};
use super::IsZeroChip;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
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
        let pis = builder.public_values();

        let x = pis[0];
        let is_zero = pis[1];

        let local = main.row_slice(0);
        let local: &IsZeroCols<AB::Var> = (*local).borrow();

        builder.assert_eq(local.x, x);
        builder.assert_eq(local.is_zero, is_zero);

        builder.assert_eq((local.x + local.is_zero) * local.inv, AB::F::one());
        builder.assert_eq(local.is_zero * local.is_zero, local.is_zero);
        builder.assert_eq(local.x * local.is_zero, AB::F::zero());
    }
}
