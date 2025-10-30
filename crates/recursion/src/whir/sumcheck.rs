use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, WhirOpeningPointBus, WhirOpeningPointMessage},
    system::Preflight,
    whir::bus::{
        WhirAlphaBus, WhirAlphaMessage, WhirEqAlphaUBus, WhirEqAlphaUMessage, WhirSumcheckBus,
        WhirSumcheckBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct SumcheckCols<T> {
    is_valid: T,
    is_first_in_group: T,
    is_last: T,
    proof_idx: T,
    tidx: T,
    sumcheck_idx: T,
    ev1: [T; D_EF],
    ev2: [T; D_EF],
    alpha: [T; D_EF],
    pre_claim: [T; D_EF],
    post_claim: [T; D_EF],
    eq_partial: [T; D_EF],
    alpha_lookup_count: T,
    whir_opening_point_msg: WhirOpeningPointMessage<T>,
    has_whir_opening_point_msg: T,
}

pub struct SumcheckAir {
    pub sumcheck_bus: WhirSumcheckBus,
    pub alpha_bus: WhirAlphaBus,
    pub eq_alpha_u_bus: WhirEqAlphaUBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for SumcheckAir {}
impl PartitionedBaseAir<F> for SumcheckAir {}

impl<F> BaseAir<F> for SumcheckAir {
    fn width(&self) -> usize {
        SumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for SumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &SumcheckCols<AB::Var> = (*local).borrow();

        self.sumcheck_bus.receive(
            builder,
            local.proof_idx,
            WhirSumcheckBusMessage {
                tidx: local.tidx,
                sumcheck_idx: local.sumcheck_idx,
                pre_claim: local.pre_claim,
                post_claim: local.post_claim,
            },
            local.is_first_in_group,
        );

        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.ev1,
            local.is_valid,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(D_EF),
            local.ev2,
            local.is_valid,
        );
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(2 * D_EF),
            local.alpha,
            local.is_valid,
        );

        self.alpha_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            WhirAlphaMessage {
                idx: local.sumcheck_idx,
                challenge: local.alpha,
            },
            local.alpha_lookup_count,
        );
        self.eq_alpha_u_bus.send(
            builder,
            local.proof_idx,
            WhirEqAlphaUMessage {
                value: local.eq_partial,
            },
            local.is_last,
        );
        self.whir_opening_point_bus.receive(
            builder,
            local.proof_idx,
            local.whir_opening_point_msg.clone(),
            local.has_whir_opening_point_msg,
        );
    }
}

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let params = vk.inner.params;
    let num_sumcheck_rounds = params.n_stack + params.l_skip - params.log_final_poly_len;
    let num_rows_per_proof = num_sumcheck_rounds;

    let num_valid_rows = num_rows_per_proof * proofs.len();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = SumcheckCols::<F>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    let mut whir_opening_point_msgs = preflights.iter().flat_map(|preflight| {
        preflight
            .whir_opening_point_messages(params.l_skip)
            .into_iter()
            .take(num_sumcheck_rounds)
    });

    for (row_idx, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let proof_idx = row_idx / num_rows_per_proof;
        let i = row_idx % num_rows_per_proof;

        let proof = &proofs[proof_idx];
        let preflight = &preflights[proof_idx];

        let cols: &mut SumcheckCols<F> = row.borrow_mut();
        cols.proof_idx = F::from_canonical_usize(proof_idx);
        cols.is_valid = F::ONE;
        let whir_round = i / vk.inner.params.k_whir;
        let j = i % vk.inner.params.k_whir;

        if let Some(msg) = whir_opening_point_msgs.next() {
            cols.whir_opening_point_msg = msg;
            cols.has_whir_opening_point_msg = F::ONE;
        }
        cols.is_first_in_group = F::from_bool(j == 0);
        cols.is_last = F::from_bool(i == num_valid_rows - 1);
        cols.sumcheck_idx = F::from_canonical_usize(i);
        cols.tidx =
            F::from_canonical_usize(preflight.whir.tidx_per_round[whir_round] + 3 * D_EF * j);
        cols.ev1 = proof.whir_proof.whir_sumcheck_polys[i][0]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.ev2 = proof.whir_proof.whir_sumcheck_polys[i][1]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.eq_partial = preflight.whir.eq_partials[i]
            .as_base_slice()
            .try_into()
            .unwrap();
        cols.alpha = preflight.whir.alphas[i].as_base_slice().try_into().unwrap();
        cols.pre_claim = if i.is_multiple_of(params.k_whir) {
            preflight.whir.initial_claim_per_round[i / params.k_whir]
                .as_base_slice()
                .try_into()
                .unwrap()
        } else {
            preflight.whir.post_sumcheck_claims[i - 1]
                .as_base_slice()
                .try_into()
                .unwrap()
        };
        cols.post_claim = preflight.whir.post_sumcheck_claims[whir_round]
            .as_base_slice()
            .try_into()
            .unwrap();
        // folding bus will do num_queries * (1 << (k_whir - j - 1)) lookups
        cols.alpha_lookup_count =
            F::from_canonical_usize(1 + (params.num_whir_queries << (params.k_whir - j - 1)));
    }

    RowMajorMatrix::new(trace, width)
}
