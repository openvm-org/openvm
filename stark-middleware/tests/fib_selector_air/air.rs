use std::borrow::Borrow;

use super::columns::FibonacciSelectorCols;
use crate::fib_air::columns::{FibonacciCols, NUM_FIBONACCI_COLS};
use afs_middleware::interaction::Chip;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

pub struct FibonacciSelectorAir {
    pub sels: Vec<bool>,
}

// No interactions
impl<F: Field> Chip<F> for FibonacciSelectorAir {}

impl<F: Field> BaseAir<F> for FibonacciSelectorAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let sels = self.sels.iter().map(|&s| F::from_bool(s)).collect();
        Some(RowMajorMatrix::new_col(sels))
    }
}

impl<AB: AirBuilderWithPublicValues + PairBuilder> Air<AB> for FibonacciSelectorAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let prep = builder.preprocessed();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let prep_local = prep.row_slice(0);
        let prep_local: &FibonacciSelectorCols<AB::Var> = (*prep_local).borrow();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &FibonacciCols<AB::Var> = (*local).borrow();
        let next: &FibonacciCols<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left, a);
        when_first_row.assert_eq(local.right, b);

        let mut when_transition = builder.when_transition();
        let mut when_selector = when_transition.when(prep_local.sel);

        // a' <- b
        when_selector.assert_eq(local.right, next.left);

        // b' <- a + b
        when_selector.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}
