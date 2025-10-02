use core::{
    borrow::{Borrow, BorrowMut},
    cmp::max,
};

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
        BatchConstraintModuleBus, BatchConstraintModuleMessage, GkrModuleBus, GkrModuleMessage,
        GkrRandomnessBus, GkrRandomnessMessage, InitialZerocheckRandomnessBus,
        InitialZerocheckRandomnessMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct DummyGkrRoundCols<T> {
    is_first: T,
    is_valid: T,
    gkr_tidx: T,
    bc_tidx: T,
    n_logup: T,
    n_max: T,
    idx: T,
    send_zc_rnd: T,
    send_gkr_rnd: T,
    challenge: [T; D_EF],
    numer_claim: [T; D_EF],
    denom_claim: [T; D_EF],
}

pub struct DummyGkrRoundAir {
    pub gkr_bus: GkrModuleBus,
    pub bc_module_bus: BatchConstraintModuleBus,
    pub initial_zc_rnd_bus: InitialZerocheckRandomnessBus,
    pub gkr_randomness_bus: GkrRandomnessBus,
}

impl BaseAirWithPublicValues<F> for DummyGkrRoundAir {}
impl PartitionedBaseAir<F> for DummyGkrRoundAir {}

impl<F> BaseAir<F> for DummyGkrRoundAir {
    fn width(&self) -> usize {
        DummyGkrRoundCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyGkrRoundAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &DummyGkrRoundCols<AB::Var> = (*local).borrow();
        let next: &DummyGkrRoundCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_one(local.is_valid);
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);

        self.gkr_bus.receive(
            builder,
            GkrModuleMessage {
                tidx: local.gkr_tidx,
                n_logup: local.n_logup,
                n_max: local.n_max,
            },
            local.is_first,
        );
        self.bc_module_bus.send(
            builder,
            BatchConstraintModuleMessage {
                tidx: local.bc_tidx,
                alpha_beta_tidx: local.gkr_tidx,
                n_max: local.n_max,
                gkr_input_layer_claim: [local.numer_claim, local.denom_claim],
            },
            local.is_first,
        );
        self.initial_zc_rnd_bus.send(
            builder,
            InitialZerocheckRandomnessMessage {
                idx: local.idx,
                challenge: local.challenge,
            },
            local.send_zc_rnd,
        );
        self.gkr_randomness_bus.send(
            builder,
            GkrRandomnessMessage {
                idx: local.idx,
                layer: local.n_logup,
                challenge: local.challenge,
            },
            local.send_gkr_rnd,
        );

        // TODO: transcript
    }
}

pub(crate) fn generate_trace(proof: &Proof, preflight: &Preflight) -> RowMajorMatrix<F> {
    let n_logup = proof.gkr_proof.claims_per_layer.len();
    let n_max = preflight.proof_shape.n_max;
    let num_valid_rows = max(n_logup, n_max + 1);

    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyGkrRoundCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut DummyGkrRoundCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(i == 0);
        cols.n_logup = F::from_canonical_usize(n_logup);
        cols.n_max = F::from_canonical_usize(n_max);
        cols.gkr_tidx = F::from_canonical_usize(preflight.proof_shape.post_tidx);
        cols.bc_tidx = F::from_canonical_usize(preflight.gkr.post_tidx);
        cols.idx = F::from_canonical_usize(i);
        cols.send_zc_rnd = F::from_bool(num_valid_rows - i - 1 <= n_max + 1);
        cols.send_gkr_rnd = F::from_bool(i < n_logup);

        cols.numer_claim
            .copy_from_slice(preflight.gkr.input_layer_numerator_claim.as_base_slice());
        cols.denom_claim
            .copy_from_slice(preflight.gkr.input_layer_denominator_claim.as_base_slice());
    }

    RowMajorMatrix::new(trace, width)
}
