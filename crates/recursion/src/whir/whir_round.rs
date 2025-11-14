use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::SubAir;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, extension::BinomiallyExtendable};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    D_EF, DIGEST_SIZE, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{CommitmentsBus, CommitmentsBusMessage, TranscriptBus, WhirModuleBus, WhirModuleMessage},
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
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
    is_enabled: T,
    proof_idx: T,
    whir_round: T,
    is_first_in_proof: T,
    tidx: T,
    z0: [T; D_EF],
    y0: [T; D_EF],
    commit: [T; DIGEST_SIZE],
    final_poly_mle_eval: [T; D_EF],
    pow_witness: T,
    pow_sample: T,
    gamma: [T; D_EF],
    // TODO: This doesn't really belong here; it should be sent directly to InitialOpenedValuesAir.
    mu: [T; D_EF],
    claim: [T; D_EF],
    next_claim: [T; D_EF],
    post_sumcheck_claim: [T; D_EF],
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
    pub num_rounds: usize,
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

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for WhirRoundAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &WhirRoundCols<AB::Var> = (*local).borrow();
        let next: &WhirRoundCols<AB::Var> = (*next).borrow();

        let proof_idx = local.proof_idx;
        let is_enabled = local.is_enabled;
        let is_proof_start = local.is_first_in_proof;
        builder.assert_bool(is_enabled);
        builder.when(is_proof_start).assert_one(is_enabled);

        let is_same_proof = next.is_enabled - next.is_first_in_proof;

        NestedForLoopSubAir.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled,
                        counter: [local.proof_idx, local.whir_round],
                        is_first: [local.is_first_in_proof, is_enabled],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_enabled,
                        counter: [next.proof_idx, next.whir_round],
                        is_first: [next.is_first_in_proof, next.is_enabled],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols {
                    is_transition: [is_same_proof.clone()],
                },
            ),
        );

        builder
            .when(local.is_enabled - is_same_proof.clone())
            .assert_eq(
                local.whir_round,
                AB::Expr::from_canonical_usize(self.num_rounds - 1),
            );

        self.whir_module_bus.receive(
            builder,
            proof_idx,
            WhirModuleMessage {
                tidx: local.tidx,
                mu: local.mu,
                claim: local.claim,
            },
            is_proof_start,
        );
        self.commitments_bus.send(
            builder,
            proof_idx,
            CommitmentsBusMessage {
                major_idx: local.whir_round + AB::Expr::ONE,
                minor_idx: AB::Expr::ZERO,
                commitment: local.commit.map(Into::into),
            },
            is_same_proof.clone() * AB::Expr::from_canonical_usize(self.num_queries),
        );

        self.sumcheck_bus.send(
            builder,
            proof_idx,
            WhirSumcheckBusMessage {
                tidx: local.tidx.into(),
                sumcheck_idx: local.whir_round * AB::Expr::from_canonical_usize(self.k),
                pre_claim: local.claim.map(Into::into),
                post_claim: local.post_sumcheck_claim.map(Into::into),
            },
            is_enabled,
        );

        let post_sumcheck_offset = 3 * self.k * D_EF;
        let mut non_final_round_offset = post_sumcheck_offset;

        self.transcript_bus.observe_commit(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(non_final_round_offset),
            local.commit,
            is_same_proof.clone(),
        );
        non_final_round_offset += DIGEST_SIZE;

        self.transcript_bus.sample_ext(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(non_final_round_offset),
            local.z0,
            is_same_proof.clone(),
        );
        non_final_round_offset += D_EF;
        self.transcript_bus.observe_ext(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(non_final_round_offset),
            local.y0,
            is_same_proof.clone(),
        );
        non_final_round_offset += D_EF;
        let is_same_proof_for_query = is_same_proof.clone();
        self.query_bus.send(
            builder,
            proof_idx,
            WhirQueryBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: AB::Expr::ZERO,
                value: local.z0.map(|z0_i| z0_i * is_same_proof_for_query.clone()),
            },
            is_enabled,
        );
        self.final_poly_mle_eval_bus.send(
            builder,
            proof_idx,
            FinalPolyMleEvalMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(post_sumcheck_offset),
                num_whir_rounds: local.whir_round + AB::Expr::ONE,
                value: local.final_poly_mle_eval.map(Into::into),
            },
            local.is_enabled - is_same_proof.clone(),
        );
        self.final_poly_query_eval_bus.send(
            builder,
            proof_idx,
            FinalPolyQueryEvalMessage {
                last_whir_round: local.whir_round + AB::Expr::ONE,
                value: ext_field_subtract(local.next_claim, local.final_poly_mle_eval),
            },
            local.is_enabled - is_same_proof.clone(),
        );

        let final_round_offset = post_sumcheck_offset + D_EF * self.final_poly_len;
        let pow_tidx = local.tidx
            + (local.is_enabled - is_same_proof.clone())
                * AB::Expr::from_canonical_usize(final_round_offset)
            + is_same_proof.clone() * AB::Expr::from_canonical_usize(non_final_round_offset);

        self.transcript_bus.observe(
            builder,
            proof_idx,
            pow_tidx.clone(),
            local.pow_witness,
            is_enabled,
        );
        self.transcript_bus.sample(
            builder,
            proof_idx,
            pow_tidx.clone() + AB::Expr::ONE,
            local.pow_sample,
            is_enabled,
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
            is_enabled,
        );

        let verify_query_tidx = pow_tidx.clone() + AB::Expr::TWO;

        self.verify_queries_bus.send(
            builder,
            proof_idx,
            VerifyQueriesBusMessage {
                tidx: verify_query_tidx,
                whir_round: local.whir_round.into(),
                gamma: local.gamma.map(Into::into),
                pre_claim: ext_field_add(
                    local.post_sumcheck_claim,
                    ext_field_multiply(local.gamma, local.y0),
                ),
                post_claim: local.next_claim.map(Into::into),
            },
            is_enabled,
        );

        self.transcript_bus.sample_ext(
            builder,
            proof_idx,
            pow_tidx.clone() + AB::Expr::from_canonical_usize(2 + self.num_queries),
            local.gamma,
            is_enabled,
        );
        builder.when(is_same_proof).assert_eq(
            next.tidx,
            pow_tidx.clone() + AB::Expr::from_canonical_usize(2 + D_EF + self.num_queries),
        );
        self.gamma_bus.send(
            builder,
            proof_idx,
            WhirGammaMessage {
                idx: local.whir_round,
                challenge: local.gamma,
            },
            is_enabled,
        );
    }
}

#[tracing::instrument(name = "generate_trace(WhirRoundAir)", skip_all)]
pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    proofs: &[&Proof],
    preflights: &[&Preflight],
) -> RowMajorMatrix<F> {
    debug_assert_eq!(proofs.len(), preflights.len());

    let params = mvk.inner.params;
    let k_whir = params.k_whir;

    let rows_per_proof = params.num_whir_rounds();
    let total_valid_rows = rows_per_proof * proofs.len();

    let height = total_valid_rows.next_power_of_two();
    let width = WhirRoundCols::<F>::width();
    let mut trace = F::zero_vec(width * height);

    trace
        .par_chunks_exact_mut(width)
        .take(total_valid_rows)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let proof_idx = row_idx / rows_per_proof;
            let i = row_idx % rows_per_proof;

            let proof = &proofs[proof_idx];
            let preflight = &preflights[proof_idx];
            let whir = &preflight.whir;
            let whir_proof = &proof.whir_proof;

            let final_poly_eval =
                whir.final_poly_at_u * *whir.eq_partials.last().expect("eq partials non-empty");

            let cols: &mut WhirRoundCols<F> = row.borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_canonical_usize(proof_idx);
            cols.is_first_in_proof = F::from_bool(i == 0);
            cols.whir_round = F::from_canonical_usize(i);
            cols.tidx = F::from_canonical_usize(whir.tidx_per_round[i]);
            cols.mu.copy_from_slice(
                preflight
                    .stacking
                    .stacking_batching_challenge
                    .as_base_slice(),
            );
            cols.claim
                .copy_from_slice(whir.initial_claim_per_round[i].as_base_slice());
            cols.final_poly_mle_eval
                .copy_from_slice(final_poly_eval.as_base_slice());

            cols.next_claim
                .copy_from_slice(whir.initial_claim_per_round[i + 1].as_base_slice());
            let post_sumcheck_idx = if k_whir == 0 {
                whir.post_sumcheck_claims.len() - 1
            } else {
                (i + 1) * k_whir - 1
            };
            cols.post_sumcheck_claim
                .copy_from_slice(whir.post_sumcheck_claims[post_sumcheck_idx].as_base_slice());
            cols.gamma.copy_from_slice(whir.gammas[i].as_base_slice());
            cols.pow_witness = whir_proof.whir_pow_witnesses[i];
            cols.pow_sample = whir.pow_samples[i];

            if i < rows_per_proof - 1 {
                cols.commit = whir_proof.codeword_commits[i];
                cols.z0.copy_from_slice(whir.z0s[i].as_base_slice());
                cols.y0
                    .copy_from_slice(whir_proof.ood_values[i].as_base_slice());
            }
        });

    RowMajorMatrix::new(trace, width)
}
