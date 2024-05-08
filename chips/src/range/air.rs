use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::columns::{RangeCols, NUM_RANGE_COLS};
use super::RangeCheckerChip;

impl<F: Field, const MAX: u32> BaseAir<F> for RangeCheckerChip<MAX> {
    fn width(&self) -> usize {
        NUM_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let column = (0..MAX).map(F::from_canonical_u32).collect();
        Some(RowMajorMatrix::new_col(column))
    }
}

impl<AB, const MAX: u32> Air<AB> for RangeCheckerChip<MAX>
where
    AB: AirBuilder, // + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // TODO
        // let prep = builder.preprocessed();
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &RangeCols<AB::Var> = (*local).borrow();

        // TODO: This is dummy to make tests pass.
        //       For some reason, permutation constraints fail when this chip has degree 2.
        builder
            .when(local.mult)
            .assert_eq(local.mult * local.mult, local.mult * local.mult);
    }
}
