use std::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_field::{AbstractField, Field},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use openvm_stark_sdk::dummy_airs::fib_air::columns::{FibonacciCols, NUM_FIBONACCI_COLS};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::columns::FibonacciSelectorCols;

pub struct FibonacciSelectorAir {
    sels: Vec<bool>,
    enable_interactions: bool,
}

impl FibonacciSelectorAir {
    pub fn new(sels: Vec<bool>, enable_interactions: bool) -> Self {
        Self {
            sels,
            enable_interactions,
        }
    }

    pub fn sels(&self) -> &[bool] {
        &self.sels
    }
}

impl<F: Field> PartitionedBaseAir<F> for FibonacciSelectorAir {}
impl<F: Field> BaseAir<F> for FibonacciSelectorAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let sels = self.sels.iter().map(|&s| F::from_bool(s)).collect();
        Some(RowMajorMatrix::new_col(sels))
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FibonacciSelectorAir {
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilderWithPublicValues + PairBuilder + InteractionBuilder> Air<AB>
    for FibonacciSelectorAir
{
    fn eval(&self, builder: &mut AB) {
        let pis = builder.public_values();
        let preprocessed = builder.preprocessed();
        let main = builder.main();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let preprocessed_local = preprocessed.row_slice(0);
        let preprocessed_local: &FibonacciSelectorCols<AB::Var> = (*preprocessed_local).borrow();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &FibonacciCols<AB::Var> = (*local).borrow();
        let next: &FibonacciCols<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left, a);
        when_first_row.assert_eq(local.right, b);

        // a' <- sel*b + (1 - sel)*a
        builder
            .when_transition()
            .when(preprocessed_local.sel)
            .assert_eq(local.right, next.left);
        builder
            .when_transition()
            .when_ne(preprocessed_local.sel, AB::Expr::ONE)
            .assert_eq(local.left, next.left);

        // b' <- sel*(a + b) + (1 - sel)*b
        builder
            .when_transition()
            .when(preprocessed_local.sel)
            .assert_eq(local.left + local.right, next.right);
        builder
            .when_transition()
            .when_ne(preprocessed_local.sel, AB::Expr::ONE)
            .assert_eq(local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);

        if self.enable_interactions {
            builder.push_receive(0, vec![local.left + local.right], preprocessed_local.sel);
        }
    }
}
