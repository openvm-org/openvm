use std::borrow::Borrow;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::PageReadChip;

impl<F: Field> BaseAir<F> for PageReadChip {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for PageReadChip
where
    AB::Var: Clone,
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // Choosing the second partition of the trace, which looks like (index, mult)
        let main: &<AB as AirBuilder>::M = &builder.partitioned_main()[1].clone();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        // Ensuring index starts at 0
        builder
            .when_first_row()
            .assert_eq(local[0], AB::Expr::zero());

        // Ensuring that index goes up by 1 every row
        builder
            .when_transition()
            .assert_eq(local[0] + AB::Expr::one(), next[0]);
    }
}
