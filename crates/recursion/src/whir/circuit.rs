use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{DIGEST_SIZE, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        StackingClaimsBus, StackingCommitmentsBus, StackingCommitmentsBusMessage,
        StackingWidthBusMessage, StackingWidthsBus,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct WhirCols<T> {
    is_first: T,
    is_valid: T,
    tidx: T,
    commit: [T; DIGEST_SIZE],
    common_stacked_width: T,
}

// Temporary dummy AIR to represent this module.
pub struct WhirAir {
    pub stacking_widths_bus: StackingWidthsBus,
    pub stacking_claims_bus: StackingClaimsBus,
    pub stacking_commitments_bus: StackingCommitmentsBus,
}

impl BaseAirWithPublicValues<F> for WhirAir {}
impl PartitionedBaseAir<F> for WhirAir {}

impl<F> BaseAir<F> for WhirAir {
    fn width(&self) -> usize {
        WhirCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for WhirAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &WhirCols<AB::Var> = (*local).borrow();
        let next: &WhirCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_one(local.is_valid);
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);

        self.stacking_commitments_bus.receive(
            builder,
            StackingCommitmentsBusMessage {
                commit_idx: AB::Expr::ZERO,
                commitment: local.commit.map(Into::into),
            },
            local.is_first,
        );
        self.stacking_widths_bus.receive(
            builder,
            StackingWidthBusMessage {
                commit_idx: AB::Expr::ZERO,
                width: local.common_stacked_width.into(),
            },
            local.is_first,
        );
    }
}

pub(crate) fn generate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let num_valid_rows: usize = 1;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = WhirCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut WhirCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(i == 0);
        cols.tidx = F::from_canonical_usize(preflight.whir_tidx);
        cols.commit = proof.common_main_commit;
        cols.common_stacked_width = F::from_canonical_usize(preflight.stacked_common_width);
    }

    RowMajorMatrix::new(trace, width)
}
