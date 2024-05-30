use std::borrow::Borrow;

use super::columns::{IsEqualCols, NUM_COLS};
use super::IsEqualChip;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::AbstractField;
// use p3_field::Field;
use p3_matrix::Matrix;

impl<F> BaseAir<F> for IsEqualChip {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for IsEqualChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let x = pis[0];
        let y = pis[1];
        let is_equal = pis[2];

        let local = main.row_slice(0);
        let local: &IsEqualCols<AB::Var> = (*local).borrow();

        builder.assert_eq(local.x, x);
        builder.assert_eq(local.y, y);
        builder.assert_eq(local.is_equal, is_equal);

        builder.assert_eq(
            (local.x - local.y + local.is_equal) * local.inv,
            AB::F::one(),
        );
        builder.assert_eq(local.is_equal * local.is_equal, local.is_equal);
        builder.assert_eq((local.x - local.y) * local.is_equal, AB::F::zero());
    }
}
