use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    bus::VariableRangeCheckerBus,
    columns::{
        VariableRangeCols, VariableRangePreprocessedCols, NUM_VARIABLE_RANGE_COLS,
        NUM_VARIABLE_RANGE_PREPROCESSED_COLS,
    },
};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct VariableRangeCheckerAir {
    pub bus: VariableRangeCheckerBus,
}

impl VariableRangeCheckerAir {
    pub fn range_max_bits(&self) -> usize {
        self.bus.range_max_bits
    }
}

impl<F: Field> BaseAir<F> for VariableRangeCheckerAir {
    fn width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows: Vec<F> = [F::zero(); NUM_VARIABLE_RANGE_PREPROCESSED_COLS]
            .into_iter()
            .chain((0..=self.range_max_bits()).flat_map(|bits| {
                (0..(1 << bits)).flat_map(move |value| {
                    [F::from_canonical_u32(value), F::from_canonical_usize(bits)].into_iter()
                })
            }))
            .collect();
        Some(RowMajorMatrix::new(
            rows,
            NUM_VARIABLE_RANGE_PREPROCESSED_COLS,
        ))
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for VariableRangeCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local: &VariableRangePreprocessedCols<AB::Var> = (*prep_local).borrow();
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &VariableRangeCols<AB::Var> = (*local).borrow();
        // Omit creating separate bridge.rs file for brevity
        self.bus
            .receive(prep_local.value, prep_local.max_bits)
            .eval(builder, local.mult);
    }
}
