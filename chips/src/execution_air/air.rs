use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::sub_chip::AirConfig;

use super::{columns::ExecutionCols, ExecutionAir};

impl<F: Field> BaseAir<F> for ExecutionAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for ExecutionAir {
    type Cols<T> = ExecutionCols<T>;
}

impl<AB: InteractionBuilder> Air<AB> for ExecutionAir {
    fn eval(&self, builder: &mut AB) {
        let main: &<AB as AirBuilder>::M = &builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();
        let local = ExecutionCols::from_slice(local, self.idx_len, self.data_len);
        let next = ExecutionCols::from_slice(next, self.idx_len, self.data_len);
        // We set the first clk to be equal to mult - this means the first op sent has clk 1.
        builder.when_first_row().assert_eq(local.mult, local.clk);
        builder.assert_bool(local.mult);
        builder.assert_zero(
            local.op_type * (local.op_type - AB::Expr::one()) * (local.op_type - AB::Expr::two()),
        );
        // clk goes up by 1 when mult is 1
        builder
            .when_transition()
            .assert_eq(next.clk, local.clk + next.mult);

        self.eval_interactions(builder, local);
    }
}
