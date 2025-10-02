use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        StackingSumcheckRandomnessBus, StackingSumcheckRandomnessMessage, WhirModuleBus,
        WhirModuleMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct WhirSumcheckCols<T> {
    is_first: T,
    is_valid: T,
    whir_tidx: T,
    round: T,
    challenge: [T; D_EF],
}

// Temporary dummy AIR to represent this module.
pub struct WhirSumcheckAir {
    pub whir_module_bus: WhirModuleBus,
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,
}

impl BaseAirWithPublicValues<F> for WhirSumcheckAir {}
impl PartitionedBaseAir<F> for WhirSumcheckAir {}

impl<F> BaseAir<F> for WhirSumcheckAir {
    fn width(&self) -> usize {
        WhirSumcheckCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for WhirSumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &WhirSumcheckCols<AB::Var> = (*local).borrow();
        let next: &WhirSumcheckCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_one(local.is_valid);
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);

        self.stacking_randomness_bus.receive(
            builder,
            StackingSumcheckRandomnessMessage {
                idx: local.round,
                challenge: local.challenge,
            },
            local.is_valid,
        );
        self.whir_module_bus.receive(
            builder,
            WhirModuleMessage {
                tidx: local.whir_tidx.into(),
            },
            local.is_first,
        );
    }
}

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let vk = &vk.inner;

    let num_valid_rows: usize = 1 + vk.params.n_stack;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = WhirSumcheckCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut WhirSumcheckCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(i == 0);
        cols.whir_tidx = F::from_canonical_usize(preflight.stacking.post_tidx);
        cols.round = F::from_canonical_usize(i);
    }

    RowMajorMatrix::new(trace, width)
}
