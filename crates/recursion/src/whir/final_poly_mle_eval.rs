use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, extension::BinomiallyExtendable};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{StackingSumcheckRandomnessBus, StackingSumcheckRandomnessMessage, TranscriptBus},
    system::Preflight,
    utils::ext_field_multiply,
    whir::bus::{
        FinalPolyMleEvalBus, FinalPolyMleEvalMessage, WhirEqAlphaUBus, WhirEqAlphaUMessage,
        WhirFinalPolyBus, WhirFinalPolyBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct FinalyPolyMleEvalCols<T> {
    is_valid: T,
    is_first: T,
    proof_idx: T,
    stacking_randomness_msg: StackingSumcheckRandomnessMessage<T>,
    has_stacking_randomness_msg: T,
    tidx: T,
    idx: T,
    coeff: [T; D_EF],
    result: [T; D_EF],
    eq_alpha_u: [T; D_EF],
    last_whir_round: T,
}

pub struct FinalPoleMleEvalAir {
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,
    pub final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    pub transcript_bus: TranscriptBus,
    pub eq_alpha_u_bus: WhirEqAlphaUBus,
    pub final_poly_bus: WhirFinalPolyBus,
}

impl BaseAirWithPublicValues<F> for FinalPoleMleEvalAir {}
impl PartitionedBaseAir<F> for FinalPoleMleEvalAir {}

impl<F> BaseAir<F> for FinalPoleMleEvalAir {
    fn width(&self) -> usize {
        FinalyPolyMleEvalCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FinalPoleMleEvalAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &FinalyPolyMleEvalCols<AB::Var> = (*local).borrow();

        self.stacking_randomness_bus.receive(
            builder,
            local.proof_idx,
            local.stacking_randomness_msg.clone(),
            local.has_stacking_randomness_msg,
        );
        self.final_poly_mle_eval_bus.receive(
            builder,
            local.proof_idx,
            FinalPolyMleEvalMessage {
                tidx: local.tidx.into(),
                last_whir_round: local.last_whir_round.into(),
                value: ext_field_multiply(local.result, local.eq_alpha_u),
            },
            local.is_first,
        );
        self.eq_alpha_u_bus.receive(
            builder,
            local.proof_idx,
            WhirEqAlphaUMessage {
                value: local.eq_alpha_u,
            },
            local.is_first,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.coeff,
            local.is_valid,
        );
        self.final_poly_bus.send(
            builder,
            local.proof_idx,
            WhirFinalPolyBusMessage {
                idx: local.idx,
                coeff: local.coeff,
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let params = vk.inner.params;

    let num_sumcheck_rounds = params.n_stack + params.l_skip - params.log_final_poly_len;
    let mut stacking_randomness_msgs = preflight
        .stacking_randomness_msgs()
        .into_iter()
        .skip(num_sumcheck_rounds);

    let num_valid_rows: usize = 1 << params.log_final_poly_len;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = FinalyPolyMleEvalCols::<F>::width();

    let mut trace = vec![F::ZERO; num_rows * width];
    let tidx = preflight.whir.tidx_per_round.last().unwrap() + 3 * params.k_whir * D_EF; // skip sumcheck

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut FinalyPolyMleEvalCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(i == 0);

        cols.tidx = F::from_canonical_usize(tidx + D_EF * i);
        cols.idx = F::from_canonical_usize(i);
        cols.coeff = proof.whir_proof.final_poly[i]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.last_whir_round = F::from_canonical_usize(preflight.whir.gammas.len());
        cols.eq_alpha_u = preflight
            .whir
            .eq_partials
            .last()
            .unwrap()
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.result = preflight
            .whir
            .final_poly_at_u
            .as_base_slice()
            .try_into()
            .unwrap();

        if let Some(msg) = stacking_randomness_msgs.next() {
            cols.stacking_randomness_msg = msg;
            cols.has_stacking_randomness_msg = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
