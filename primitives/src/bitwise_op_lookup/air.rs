use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    bus::BitwiseOperationLookupBus,
    columns::{
        BitwiseOperationLookupCols, BitwiseOperationLookupPreprocessedCols,
        NUM_BITWISE_OP_LOOKUP_COLS, NUM_BITWISE_OP_LOOKUP_PREPROCESSED_COLS,
    },
};

#[derive(Copy, Clone, PartialEq)]
pub enum BitwiseOperationLookupOpcode {
    ADD = 0,
    XOR = 1,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct BitwiseOperationLookupAir<const NUM_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
}

impl<F: Field, const NUM_BITS: usize> BaseAir<F> for BitwiseOperationLookupAir<NUM_BITS> {
    fn width(&self) -> usize {
        NUM_BITWISE_OP_LOOKUP_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows: Vec<F> = (0..(1 << NUM_BITS))
            .flat_map(|x: u32| {
                (0..(1 << NUM_BITS)).flat_map(move |y: u32| {
                    [
                        F::from_canonical_u32(x),
                        F::from_canonical_u32(y),
                        F::from_canonical_u32((x + y) % (1 << NUM_BITS)),
                        F::from_canonical_u32(x ^ y),
                    ]
                })
            })
            .collect();
        Some(RowMajorMatrix::new(
            rows,
            NUM_BITWISE_OP_LOOKUP_PREPROCESSED_COLS,
        ))
    }
}

impl<AB: InteractionBuilder + PairBuilder, const NUM_BITS: usize> Air<AB>
    for BitwiseOperationLookupAir<NUM_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local: &BitwiseOperationLookupPreprocessedCols<AB::Var> = (*prep_local).borrow();

        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BitwiseOperationLookupCols<AB::Var> = (*local).borrow();

        self.bus
            .receive(
                prep_local.x,
                prep_local.y,
                prep_local.z_add,
                AB::Expr::from_canonical_u8(BitwiseOperationLookupOpcode::ADD as u8),
            )
            .eval(builder, local.mult_add);
        self.bus
            .receive(
                prep_local.x,
                prep_local.y,
                prep_local.z_xor,
                AB::Expr::from_canonical_u8(BitwiseOperationLookupOpcode::XOR as u8),
            )
            .eval(builder, local.mult_xor);
    }
}
