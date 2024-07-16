use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::sub_chip::SubAirBridge;

use super::columns::NUM_RANGE_COLS;
use super::RangeCheckerAir;

impl<F: Field> BaseAir<F> for RangeCheckerAir {
    fn width(&self) -> usize {
        NUM_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let column = (0..self.range_max).map(F::from_canonical_u32).collect();
        Some(RowMajorMatrix::new_col(column))
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for RangeCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local = (*prep_local).borrow();
        let main = builder.main();
        let local = main.row_slice(0);
        let local = (*local).borrow();
        self.eval_interactions(builder, (*prep_local, *local));
    }
}
