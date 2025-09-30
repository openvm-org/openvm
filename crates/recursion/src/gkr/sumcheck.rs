use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::bus::{GkrRandomnessBus, GkrRandomnessMessage};

#[repr(C)]
#[derive(AlignedBorrow)]
struct DummyGkrSumcheckCols<T> {
    is_valid: T,
    tidx: T,
    /// The GKR layer.
    layer: T,
    /// The sumcheck round for this layer.
    round: T,
    challenge: [T; D_EF],
}

pub struct DummyGkrSumcheckAir {
    pub gkr_randomness_bus: GkrRandomnessBus,
}

impl BaseAirWithPublicValues<F> for DummyGkrSumcheckAir {}
impl PartitionedBaseAir<F> for DummyGkrSumcheckAir {}

impl<F> BaseAir<F> for DummyGkrSumcheckAir {
    fn width(&self) -> usize {
        DummyGkrSumcheckCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyGkrSumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &DummyGkrSumcheckCols<AB::Var> = (*local).borrow();
        let next: &DummyGkrSumcheckCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        self.gkr_randomness_bus.send(
            builder,
            GkrRandomnessMessage {
                idx: local.round,
                layer: local.layer,
                challenge: local.challenge,
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace(proof: &Proof) -> RowMajorMatrix<F> {
    let num_valid_rows = proof.gkr_proof.claims_per_layer.len();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyGkrSumcheckCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut DummyGkrSumcheckCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.tidx = F::ZERO;
    }

    RowMajorMatrix::new(trace, width)
}
