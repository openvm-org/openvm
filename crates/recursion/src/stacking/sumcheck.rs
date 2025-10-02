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
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, StackingModuleBus,
        StackingModuleMessage, StackingSumcheckRandomnessBus, StackingSumcheckRandomnessMessage,
        WhirModuleBus, WhirModuleMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct StackingCols<T> {
    is_first: T,
    is_valid: T,
    stacking_tidx: T,
    whir_tidx: T,
    round: T,
    bc_rand: [T; D_EF],
    challenge: [T; D_EF],
    recv_bc: T,
}

// Temporary dummy AIR to represent this module.
pub struct StackingSumcheckAir {
    pub stacking_module_bus: StackingModuleBus,
    pub whir_module_bus: WhirModuleBus,
    // This probably belongs on another AIR, but putting here now just for balancing interactions
    pub batch_constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,
}

impl BaseAirWithPublicValues<F> for StackingSumcheckAir {}
impl PartitionedBaseAir<F> for StackingSumcheckAir {}

impl<F> BaseAir<F> for StackingSumcheckAir {
    fn width(&self) -> usize {
        StackingCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for StackingSumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &StackingCols<AB::Var> = (*local).borrow();
        let next: &StackingCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_one(local.is_valid);
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);

        self.stacking_module_bus.receive(
            builder,
            StackingModuleMessage {
                tidx: local.stacking_tidx.into(),
            },
            local.is_first,
        );
        self.batch_constraint_randomness_bus.receive(
            builder,
            ConstraintSumcheckRandomness {
                idx: local.round,
                challenge: local.bc_rand,
            },
            local.recv_bc,
        );
        self.stacking_randomness_bus.send(
            builder,
            StackingSumcheckRandomnessMessage {
                idx: local.round,
                challenge: local.challenge,
            },
            local.is_valid,
        );
        self.whir_module_bus.send(
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
    let width = StackingCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut StackingCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(i == 0);
        cols.stacking_tidx = F::from_canonical_usize(preflight.batch_constraint.post_tidx);
        cols.whir_tidx = F::from_canonical_usize(preflight.stacking.post_tidx);
        cols.round = F::from_canonical_usize(i);
        cols.recv_bc = F::from_bool(i <= preflight.proof_shape.n_max);
    }

    RowMajorMatrix::new(trace, width)
}
