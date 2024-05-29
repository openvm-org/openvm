use std::borrow::Borrow;

use super::columns::{IsZeroCols, NUM_COLS};
use afs_stark_backend::interaction::Chip;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
use p3_field::Field;
use p3_matrix::Matrix;

pub struct IsZeroAir;

// No interactions
impl<F: Field> Chip<F> for IsZeroAir {}

impl<F> BaseAir<F> for IsZeroAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for IsZeroAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let x = pis[0];
        let is_zero = pis[1];

        let local = main.row_slice(0);
        let local: &IsZeroCols<AB::Var> = (*local).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.x, x);
        when_first_row.assert_eq(local.is_zero, is_zero);

        when_first_row.assert_eq((local.x + local.is_zero) * local.inv, AB::F::one());
        when_first_row.assert_eq(local.is_zero * local.is_zero, local.is_zero);
        when_first_row.assert_eq(local.x * local.is_zero, AB::F::zero());
    }
}
