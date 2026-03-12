use core::borrow::Borrow;

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;

use crate::{
    bus::{CommitmentsBus, CommitmentsBusMessage, TranscriptBus, WhirModuleBus, WhirModuleMessage},
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract, pow_tidx_count},
    whir::bus::{
        FinalPolyMleEvalBus, FinalPolyMleEvalMessage, FinalPolyQueryEvalBus,
        FinalPolyQueryEvalMessage, VerifyQueriesBus, VerifyQueriesBusMessage, WhirGammaBus,
        WhirGammaMessage, WhirQueryBus, WhirQueryBusMessage, WhirSumcheckBus,
        WhirSumcheckBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct WhirRoundCols<T, const ENC_WIDTH: usize> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub whir_round: T,
    pub is_first_in_proof: T,
    pub tidx: T,
    pub num_queries: T,
    pub omega: T,
    pub z0: [T; D_EF],
    pub y0: [T; D_EF],
    pub commit: [T; DIGEST_SIZE],
    pub final_poly_mle_eval: [T; D_EF],
    pub query_pow_witness: T,
    pub query_pow_sample: T,
    pub gamma: [T; D_EF],
    pub claim: [T; D_EF],
    pub next_claim: [T; D_EF],
    pub post_sumcheck_claim: [T; D_EF],
    pub whir_round_enc: [T; ENC_WIDTH],
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
    pub num_rounds: usize,
    pub initial_log_domain_size: usize,
    pub final_poly_len: usize,
    pub pow_bits: usize,
    pub folding_pow_bits: usize,
    pub generator: F,

    pub whir_round_encoder: Encoder,
    pub num_queries_per_round: Vec<usize>,
}

impl BaseAirWithPublicValues<F> for WhirRoundAir {}
impl PartitionedBaseAir<F> for WhirRoundAir {}

impl<F> BaseAir<F> for WhirRoundAir {
    fn width(&self) -> usize {
        match self.whir_round_encoder.width() {
            1 => WhirRoundCols::<usize, 1>::width(),
            2 => WhirRoundCols::<usize, 2>::width(),
            3 => WhirRoundCols::<usize, 3>::width(),
            w => panic!("unsupported encoder width: {w}"),
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for WhirRoundAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        BinomiallyExtendable<{ D_EF }> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        match self.whir_round_encoder.width() {
            1 => self.eval_impl::<AB, 1>(builder),
            2 => self.eval_impl::<AB, 2>(builder),
            3 => self.eval_impl::<AB, 3>(builder),
            w => panic!("unsupported encoder width: {w}"),
        }
    }
}

impl WhirRoundAir {
    fn eval_impl<AB: AirBuilder<F = F> + InteractionBuilder, const ENC_WIDTH: usize>(
        &self,
        builder: &mut AB,
    ) where
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
            BinomiallyExtendable<{ D_EF }> + TwoAdicField,
    {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &WhirRoundCols<AB::Var, ENC_WIDTH> = (*local).borrow();
        let next: &WhirRoundCols<AB::Var, ENC_WIDTH> = (*next).borrow();

        let proof_idx = local.proof_idx;
        let is_enabled = local.is_enabled;
        let is_proof_start = local.is_first_in_proof;
        self.whir_round_encoder.eval(builder, &local.whir_round_enc);

        // Constrain whir_round column to match the decoded encoder value
        let round_flag_vals: Vec<_> = (0..self.num_rounds).map(|r| (r, r)).collect();
        let whir_round_decoded = self
            .whir_round_encoder
            .flag_with_val::<AB>(&local.whir_round_enc, &round_flag_vals);
        builder
            .when(is_enabled)
            .assert_eq(local.whir_round, whir_round_decoded);

        // Constrain num_queries column to match the expected value for this round
        let num_queries_flag_vals: Vec<_> = self
            .num_queries_per_round
            .iter()
            .enumerate()
            .map(|(r, &nq)| (r, nq))
            .collect();
        let expected_num_queries = self
            .whir_round_encoder
            .flag_with_val::<AB>(&local.whir_round_enc, &num_queries_flag_vals);
        builder
            .when(is_enabled)
            .assert_eq(local.num_queries, expected_num_queries);

        let initial_omega = AB::Expr::from_prime_subfield(
            <<AB::Expr as PrimeCharacteristicRing>::PrimeSubfield as TwoAdicField>::two_adic_generator(
                self.initial_log_domain_size,
            ),
        );
        builder
            .when(is_proof_start)
            .assert_eq(local.omega, initial_omega);

        // Use the column (degree 1) instead of decoded expression (high degree) for constraints
        let whir_round: AB::Expr = local.whir_round.into();
        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        builder
            .when(is_same_proof.clone())
            .assert_eq(next.omega, local.omega * local.omega);

        NestedForLoopSubAir.eval(
            builder,
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
        );

        builder
            .when(local.is_enabled - is_same_proof.clone())
            .assert_eq(local.whir_round, AB::Expr::from_usize(self.num_rounds - 1));

        assert_array_eq(
            &mut builder.when(is_same_proof.clone()),
            next.claim,
            local.next_claim,
        );

        self.whir_module_bus.receive(
            builder,
            proof_idx,
            WhirModuleMessage {
                tidx: local.tidx,
                claim: local.claim,
            },
            is_proof_start,
        );
        self.commitments_bus.add_key_with_lookups(
            builder,
            proof_idx,
            CommitmentsBusMessage {
                major_idx: whir_round.clone() + AB::Expr::ONE,
                minor_idx: AB::Expr::ZERO,
                commitment: local.commit.map(Into::into),
            },
            is_same_proof.clone() * next.num_queries,
        );

        self.sumcheck_bus.send(
            builder,
            proof_idx,
            WhirSumcheckBusMessage {
                tidx: local.tidx.into(),
                sumcheck_idx: whir_round.clone() * AB::Expr::from_usize(self.k),
                pre_claim: local.claim.map(Into::into),
                post_claim: local.post_sumcheck_claim.map(Into::into),
            },
            is_enabled,
        );

        let folding_pow_offset = pow_tidx_count(self.folding_pow_bits);
        let query_pow_offset = pow_tidx_count(self.pow_bits);
        let post_sumcheck_offset = (3 * D_EF + folding_pow_offset) * self.k;
        let mut non_final_round_offset = post_sumcheck_offset;

        self.transcript_bus.observe_commit(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_usize(non_final_round_offset),
            local.commit,
            is_same_proof.clone(),
        );
        non_final_round_offset += DIGEST_SIZE;

        self.transcript_bus.sample_ext(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_usize(non_final_round_offset),
            local.z0,
            is_same_proof.clone(),
        );
        non_final_round_offset += D_EF;
        self.transcript_bus.observe_ext(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_usize(non_final_round_offset),
            local.y0,
            is_same_proof.clone(),
        );
        non_final_round_offset += D_EF;
        self.query_bus.send(
            builder,
            proof_idx,
            WhirQueryBusMessage {
                whir_round: whir_round.clone(),
                query_idx: AB::Expr::ZERO,
                value: local.z0.map(|z0_i| z0_i * is_same_proof.clone()),
            },
            is_enabled,
        );
        self.final_poly_mle_eval_bus.send(
            builder,
            proof_idx,
            FinalPolyMleEvalMessage {
                tidx: local.tidx + AB::Expr::from_usize(post_sumcheck_offset),
                num_whir_rounds: whir_round.clone() + AB::Expr::ONE,
                value: local.final_poly_mle_eval.map(Into::into),
            },
            local.is_enabled - is_same_proof.clone(),
        );
        self.final_poly_query_eval_bus.send(
            builder,
            proof_idx,
            FinalPolyQueryEvalMessage {
                last_whir_round: whir_round.clone() + AB::Expr::ONE,
                value: ext_field_subtract(local.next_claim, local.final_poly_mle_eval),
            },
            local.is_enabled - is_same_proof.clone(),
        );

        let final_round_offset = post_sumcheck_offset + D_EF * self.final_poly_len;
        let pow_tidx = local.tidx
            + (local.is_enabled - is_same_proof.clone()) * AB::Expr::from_usize(final_round_offset)
            + is_same_proof.clone() * AB::Expr::from_usize(non_final_round_offset);

        if self.pow_bits > 0 {
            self.transcript_bus.observe(
                builder,
                proof_idx,
                pow_tidx.clone(),
                local.query_pow_witness,
                is_enabled,
            );
            self.transcript_bus.sample(
                builder,
                proof_idx,
                pow_tidx.clone() + AB::Expr::ONE,
                local.query_pow_sample,
                is_enabled,
            );

            // Check proof-of-work using `ExpBitsLenBus`.
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: self.generator.into(),
                    bit_src: local.query_pow_sample.into(),
                    num_bits: AB::Expr::from_usize(self.pow_bits),
                    result: AB::Expr::ONE,
                },
                is_enabled,
            );
        }

        let verify_query_tidx = pow_tidx.clone() + AB::Expr::from_usize(query_pow_offset);
        let y0_gated: [AB::Expr; D_EF] = local.y0.map(|y| y * is_same_proof.clone());

        self.verify_queries_bus.send(
            builder,
            proof_idx,
            VerifyQueriesBusMessage {
                tidx: verify_query_tidx,
                whir_round: whir_round.clone(),
                num_queries: local.num_queries.into(),
                omega: local.omega.into(),
                gamma: local.gamma.map(Into::into),
                pre_claim: ext_field_add(
                    local.post_sumcheck_claim,
                    ext_field_multiply(local.gamma, y0_gated),
                ),
                post_claim: local.next_claim.map(Into::into),
            },
            is_enabled,
        );

        self.transcript_bus.sample_ext(
            builder,
            proof_idx,
            pow_tidx.clone() + AB::Expr::from_usize(query_pow_offset) + local.num_queries,
            local.gamma,
            is_enabled,
        );
        builder.when(is_same_proof).assert_eq(
            next.tidx,
            pow_tidx.clone() + AB::Expr::from_usize(query_pow_offset + D_EF) + local.num_queries,
        );
        self.gamma_bus.send(
            builder,
            proof_idx,
            WhirGammaMessage {
                idx: whir_round.clone(),
                challenge: local.gamma.map(Into::into),
            },
            is_enabled,
        );
    }
}
