use std::borrow::Borrow;

use super::columns::{FibonacciCols, NUM_FIBONACCI_COLS};
use afs_stark_backend::interaction::AirBridge;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

pub struct FibonacciAir;

// No interactions
impl<F: Field> AirBridge<F> for FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &FibonacciCols<AB::Var> = (*local).borrow();
        let next: &FibonacciCols<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left, a);
        when_first_row.assert_eq(local.middle, b);
        when_first_row.assert_eq(local.right, local.left + local.middle);

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.middle, next.left);

        // b' <- c
        when_transition.assert_eq(local.right, next.middle);

        // c' <- b + c
        when_transition.assert_eq(local.middle + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}
