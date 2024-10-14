use std::borrow::Borrow;

use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    bus::RangeCheckBus,
    columns::{RangeCols, RangePreprocessedCols, NUM_RANGE_COLS},
};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct RangeCheckerAir {
    pub bus: RangeCheckBus,
}

impl RangeCheckerAir {
    pub fn range_max(&self) -> u32 {
        self.bus.range_max
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for RangeCheckerAir {}
impl<F: Field> PartitionedBaseAir<F> for RangeCheckerAir {}
impl<F: Field> BaseAir<F> for RangeCheckerAir {
    fn width(&self) -> usize {
        NUM_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let column = (0..self.range_max()).map(F::from_canonical_u32).collect();
        Some(RowMajorMatrix::new_col(column))
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for RangeCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local: &RangePreprocessedCols<AB::Var> = (*prep_local).borrow();
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &RangeCols<AB::Var> = (*local).borrow();
        // Omit creating separate bridge.rs file for brevity
        self.bus
            .receive(prep_local.counter)
            .eval(builder, local.mult);
    }
}
