use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::columns::{XorLookupCols, XorLookupPreprocessedCols, NUM_XOR_LOOKUP_COLS};

#[derive(Clone, Copy, Debug)]
pub struct XorLookupAir<const M: usize> {
    pub bus_index: usize,
}

impl<const M: usize> XorLookupAir<M> {
    pub fn new(bus_index: usize) -> Self {
        Self { bus_index }
    }
}

impl<F: Field, const M: usize> BaseAir<F> for XorLookupAir<M> {
    fn width(&self) -> usize {
        NUM_XOR_LOOKUP_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows: Vec<Vec<F>> = (0..(1 << M) * (1 << M))
            .map(|i| {
                let x = i / (1 << M);
                let y = i % (1 << M);
                let z = x ^ y;
                vec![
                    F::from_canonical_usize(x),
                    F::from_canonical_usize(y),
                    F::from_canonical_usize(z),
                ]
            })
            .collect();

        Some(RowMajorMatrix::new(rows.concat(), 3))
    }
}

impl<AB, const M: usize> Air<AB> for XorLookupAir<M>
where
    AB: InteractionBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();

        let prep_local = preprocessed.row_slice(0);
        let prep_local: &XorLookupPreprocessedCols<AB::Var> = (*prep_local).borrow();
        let local = main.row_slice(0);
        let local: &XorLookupCols<AB::Var> = (*local).borrow();

        self.eval_interactions(builder, *prep_local, *local);
    }
}
