use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        BatchConstraintModuleBus, BatchConstraintModuleMessage, ConstraintSumcheckRandomness,
        ConstraintSumcheckRandomnessBus, InitialZerocheckRandomnessBus,
        InitialZerocheckRandomnessMessage, StackingModuleBus, StackingModuleMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct BatchConstraintSumcheckCols<T> {
    is_first: T,
    is_last: T,
    is_valid: T,
    round: T,
    challenge: [T; D_EF],
    initial_rnd: [T; D_EF],
    bc_tidx: T,
    alpha_beta_tidx: T,
    stacking_tidx: T,
    n_max: T,
    numer_claim: [T; D_EF],
    denom_claim: [T; D_EF],
}

pub struct BatchConstraintSumcheckAir {
    pub bc_module_bus: BatchConstraintModuleBus,
    pub stacking_module_bus: StackingModuleBus,
    pub initial_zc_randomness_bus: InitialZerocheckRandomnessBus,
    pub batch_constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
}

impl BaseAirWithPublicValues<F> for BatchConstraintSumcheckAir {}
impl PartitionedBaseAir<F> for BatchConstraintSumcheckAir {}

impl<F> BaseAir<F> for BatchConstraintSumcheckAir {
    fn width(&self) -> usize {
        BatchConstraintSumcheckCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for BatchConstraintSumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &BatchConstraintSumcheckCols<AB::Var> = (*local).borrow();
        let next: &BatchConstraintSumcheckCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_one(local.is_valid);
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);

        self.bc_module_bus.receive(
            builder,
            BatchConstraintModuleMessage {
                tidx: local.bc_tidx,
                alpha_beta_tidx: local.alpha_beta_tidx,
                n_max: local.n_max,
                gkr_input_layer_claim: [local.numer_claim, local.denom_claim],
            },
            local.is_first,
        );

        self.stacking_module_bus.send(
            builder,
            StackingModuleMessage {
                tidx: local.stacking_tidx,
            },
            local.is_last,
        );
        self.initial_zc_randomness_bus.receive(
            builder,
            InitialZerocheckRandomnessMessage {
                idx: local.round,
                challenge: local.initial_rnd,
            },
            local.is_valid,
        );
        self.batch_constraint_randomness_bus.send(
            builder,
            ConstraintSumcheckRandomness {
                idx: local.round,
                challenge: local.challenge,
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace(proof: &Proof, preflight: &Preflight) -> RowMajorMatrix<F> {
    let num_valid_rows: usize = preflight.proof_shape.n_max + 1;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = BatchConstraintSumcheckCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut BatchConstraintSumcheckCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.n_max = F::from_canonical_usize(preflight.proof_shape.n_max);
        cols.bc_tidx = F::from_canonical_usize(preflight.gkr.post_tidx);
        cols.alpha_beta_tidx = F::from_canonical_usize(preflight.proof_shape.post_tidx);
        cols.stacking_tidx = F::from_canonical_usize(preflight.batch_constraint.post_tidx);
        cols.numer_claim
            .copy_from_slice(preflight.gkr.input_layer_numerator_claim.as_base_slice());
        cols.denom_claim
            .copy_from_slice(preflight.gkr.input_layer_denominator_claim.as_base_slice());
        cols.round = F::from_canonical_usize(i);
        if i == 0 {
            cols.is_first = F::ONE;
        }
        if i == num_valid_rows - 1 {
            cols.is_last = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
