use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    D_EF, DIGEST_SIZE, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        CommitmentsBus, CommitmentsBusMessage, ExpBitsLenBus, ExpBitsLenMessage, TranscriptBus,
        WhirModuleBus, WhirModuleMessage,
    },
    system::Preflight,
    utils::ext_field_subtract,
    whir::bus::{
        FinalPolyMleEvalBus, FinalPolyMleEvalMessage, FinalPolyQueryEvalBus,
        FinalPolyQueryEvalMessage, VerifyQueriesBus, VerifyQueriesBusMessage, WhirGammaBus,
        WhirGammaMessage, WhirQueryBus, WhirQueryBusMessage, WhirSumcheckBus,
        WhirSumcheckBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct WhirRoundCols<T> {
    is_valid: T,
    is_first_round: T,
    is_last_round: T,
    whir_round: T,
    tidx: T,
    proof_idx: T,
    z0: [T; D_EF],
    y0: [T; D_EF],
    commit: [T; DIGEST_SIZE],
    pow_witness: T,
    pow_sample: T,
    gamma: [T; D_EF],

    claim: [T; D_EF],
    post_sumcheck_claim: [T; D_EF],
    // TODO: Remove these column; they are quadratic functions of gamma, post_sumcheck_claim and
    // y0. (Must constrain y0 to be 0 on the final round.)
    pre_query_claim: [T; D_EF],
    // TODO: consider using next.claim and a final summary row
    post_query_claim: [T; D_EF],

    whir_module_msg: WhirModuleMessage<T>,
    has_commitments_bus_msg: T,
    commitments_bus_msg: CommitmentsBusMessage<T>,
}

pub struct WhirRoundAir {
    // extra-module buses
    pub whir_module_bus: WhirModuleBus,
    pub commitments_bus: CommitmentsBus,
    pub transcript_bus: TranscriptBus,

    // intra-module buses
    pub gamma_bus: WhirGammaBus,
    pub query_bus: WhirQueryBus,
    pub sumcheck_bus: WhirSumcheckBus,
    pub verify_queries_bus: VerifyQueriesBus,
    pub final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    pub final_poly_query_eval_bus: FinalPolyQueryEvalBus,
    pub exp_bits_len_bus: ExpBitsLenBus,

    pub k: usize,
    pub num_queries: usize,
    pub final_poly_len: usize,
    pub pow_bits: usize,
    pub generator: F,
}

impl BaseAirWithPublicValues<F> for WhirRoundAir {}
impl PartitionedBaseAir<F> for WhirRoundAir {}

impl<F> BaseAir<F> for WhirRoundAir {
    fn width(&self) -> usize {
        WhirRoundCols::<usize>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for WhirRoundAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &WhirRoundCols<AB::Var> = (*local).borrow();
        let next: &WhirRoundCols<AB::Var> = (*next).borrow();

        self.whir_module_bus.receive(
            builder,
            local.proof_idx,
            local.whir_module_msg.clone(),
            local.is_first_round,
        );
        self.commitments_bus.send(
            builder,
            local.proof_idx,
            local.commitments_bus_msg.clone(),
            local.has_commitments_bus_msg,
        );

        self.sumcheck_bus.send(
            builder,
            local.proof_idx,
            WhirSumcheckBusMessage {
                tidx: local.tidx.into(),
                sumcheck_idx: local.whir_round * AB::Expr::from_canonical_usize(self.k),
                pre_claim: local.claim.map(Into::into),
                post_claim: local.post_sumcheck_claim.map(Into::into),
            },
            local.is_valid,
        );

        let post_sumcheck_offset = 3 * self.k * D_EF;
        let mut non_final_round_offset = post_sumcheck_offset;

        self.transcript_bus.observe_commit(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(non_final_round_offset),
            local.commit,
            local.is_valid - local.is_last_round,
        );
        non_final_round_offset += DIGEST_SIZE;

        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(non_final_round_offset),
            local.z0,
            local.is_valid - local.is_last_round,
        );
        non_final_round_offset += D_EF;
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(non_final_round_offset),
            local.y0,
            local.is_valid - local.is_last_round,
        );
        non_final_round_offset += D_EF;
        self.query_bus.send(
            builder,
            local.proof_idx,
            WhirQueryBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: AB::Expr::ZERO,
                value: local.z0.map(Into::into),
            },
            // Send even on last row, where `value` is contrained to be zero.
            local.is_valid,
        );
        // On the summary row we use `y0` and `z0` as final poly MLE and query eval results,
        // respectively.
        self.final_poly_mle_eval_bus.send(
            builder,
            local.proof_idx,
            FinalPolyMleEvalMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(post_sumcheck_offset),
                last_whir_round: next.whir_round.into(),
                value: next.y0.map(Into::into),
            },
            local.is_last_round,
        );
        self.final_poly_query_eval_bus.send(
            builder,
            local.proof_idx,
            FinalPolyQueryEvalMessage {
                last_whir_round: next.whir_round.into(),
                value: ext_field_subtract(next.claim, next.y0),
            },
            local.is_last_round,
        );

        let final_round_offset = post_sumcheck_offset + D_EF * self.final_poly_len;
        let pow_tidx = local.tidx
            + local.is_last_round * AB::Expr::from_canonical_usize(final_round_offset)
            + (AB::Expr::ONE - local.is_last_round)
                * AB::Expr::from_canonical_usize(non_final_round_offset);

        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            pow_tidx.clone(),
            local.pow_witness.into(),
            local.is_valid.into(),
        );
        self.transcript_bus.sample(
            builder,
            local.proof_idx,
            pow_tidx.clone() + AB::Expr::ONE,
            local.pow_sample.into(),
            local.is_valid.into(),
        );

        // Check proof-of-work using `ExpBitsLenBus`.
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: self.generator.into(),
                bit_src: local.pow_sample.into(),
                num_bits: AB::Expr::from_canonical_usize(self.pow_bits),
                result: AB::Expr::ONE,
            },
            local.is_valid,
        );

        let verify_query_tidx = pow_tidx.clone() + AB::Expr::TWO;

        self.verify_queries_bus.send(
            builder,
            local.proof_idx,
            VerifyQueriesBusMessage {
                tidx: verify_query_tidx,
                whir_round: local.whir_round.into(),
                gamma: local.gamma.map(Into::into),
                pre_claim: local.pre_query_claim.map(Into::into),
                post_claim: local.post_query_claim.map(Into::into),
            },
            local.is_valid,
        );

        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            pow_tidx.clone() + AB::Expr::from_canonical_usize(2 + self.num_queries),
            local.gamma,
            local.is_valid.into(),
        );
        self.gamma_bus.send(
            builder,
            local.proof_idx,
            WhirGammaMessage {
                idx: local.whir_round,
                challenge: local.gamma,
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let params = mvk.inner.params;
    let rows_per_proof = params.num_whir_rounds() + 1;
    let num_valid_rows = rows_per_proof * proofs.len();
    let height = num_valid_rows.next_power_of_two();
    let width = WhirRoundCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];

    for (row_idx, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let proof_idx = row_idx / rows_per_proof;
        let i = row_idx % rows_per_proof;
        let proof = &proofs[proof_idx];
        let preflight = &preflights[proof_idx];

        let commitments_bus_msgs = preflight.whir_commitments_msgs(proof);
        let whir_proof = &proof.whir_proof;

        let cols: &mut WhirRoundCols<F> = row.borrow_mut();
        cols.proof_idx = F::from_canonical_usize(proof_idx);
        if i == 0 {
            cols.is_first_round = F::ONE;
            cols.whir_module_msg = preflight.whir_module_msg(proof);
        }
        if i < commitments_bus_msgs.len() {
            cols.has_commitments_bus_msg =
                F::from_canonical_usize(mvk.inner.params.num_whir_queries);
            cols.commitments_bus_msg = commitments_bus_msgs[i].clone();
        }
        cols.whir_round = F::from_canonical_usize(i);
        cols.claim = preflight.whir.initial_claim_per_round[i]
            .as_base_slice()
            .try_into()
            .unwrap();
        if i == rows_per_proof - 1 {
            cols.y0 = (preflight.whir.final_poly_at_u
                * *preflight.whir.eq_partials.last().unwrap())
            .as_base_slice()
            .try_into()
            .unwrap();
        } else if i < rows_per_proof - 1 {
            cols.is_valid = F::ONE;
            cols.tidx = F::from_canonical_usize(preflight.whir.tidx_per_round[i]);
            cols.is_last_round = F::from_bool(i == rows_per_proof - 2);
            if i < rows_per_proof - 2 {
                cols.commit = whir_proof.codeword_commits[i];
                cols.z0 = preflight.whir.z0s[i].as_base_slice().try_into().unwrap();
                cols.y0 = whir_proof.ood_values[i].as_base_slice().try_into().unwrap();
            }

            cols.post_sumcheck_claim = preflight.whir.post_sumcheck_claims[i]
                .as_base_slice()
                .try_into()
                .unwrap();
            cols.pre_query_claim = preflight.whir.pre_query_claims[i]
                .as_base_slice()
                .try_into()
                .unwrap();
            cols.post_query_claim = preflight.whir.initial_claim_per_round[i + 1]
                .as_base_slice()
                .try_into()
                .unwrap();
            cols.gamma = preflight.whir.gammas[i].as_base_slice().try_into().unwrap();
            cols.pow_witness = whir_proof.whir_pow_witnesses[i];
            cols.pow_sample = preflight.whir.pow_samples[i];
        }
    }
    RowMajorMatrix::new(trace, width)
}
