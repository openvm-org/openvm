use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::RangeGateCols;
use super::columns::NUM_RANGE_GATE_COLS;
use super::RangeCheckerGateChip;

impl<F: Field> BaseAir<F> for RangeCheckerGateChip {
    fn width(&self) -> usize {
        NUM_RANGE_GATE_COLS
    }
}

impl<AB> Air<AB> for RangeCheckerGateChip
where
    AB: AirBuilder,
{
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
    }
}
