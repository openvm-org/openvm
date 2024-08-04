use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{RangeGateCols, NUM_RANGE_GATE_COLS};

#[derive(Clone, Copy, Debug)]
pub struct RangeCheckerGateAir {
    pub bus_index: usize,
    pub range_max: u32,
}

impl<F: Field> BaseAir<F> for RangeCheckerGateAir {
    fn width(&self) -> usize {
        NUM_RANGE_GATE_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for RangeCheckerGateAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &RangeGateCols<AB::Var> = (*local).borrow();
        let next: &RangeGateCols<AB::Var> = (*next).borrow();

        builder
            .when_first_row()
            .assert_eq(local.counter, AB::Expr::zero());
        builder
            .when_transition()
            .assert_eq(local.counter + AB::Expr::one(), next.counter);

        self.eval_interactions(builder, local.counter, local.mult);
    }
}
