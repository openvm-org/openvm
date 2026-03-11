use std::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{and, assert_array_eq, not},
    SubAir,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;

use crate::{
    bus::{
        StackingIndexMessage, StackingIndicesBus, TranscriptBus, TranscriptBusMessage,
        WhirModuleBus, WhirModuleMessage, WhirMuBus, WhirMuMessage,
    },
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    stacking::bus::{
        ClaimCoefficientsBus, ClaimCoefficientsMessage, StackingModuleTidxBus,
        StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{assert_one_ext, ext_field_add, ext_field_multiply, pow_tidx_count},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct StackingClaimsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    /// Row has a real stacking claim (bus interactions fire).
    pub is_valid: F,
    /// Row is padding within a proof block (no bus interactions).
    pub is_padding: F,
    pub is_first: F,
    /// Last row of the proof block (valid + padding). Triggers proof_idx
    /// transition and the w_stack check.
    pub is_last: F,

    // Correspond to stacking_claim
    pub commit_idx: F,
    pub stacked_col_idx: F,

    // Sampled transcript values
    pub tidx: F,
    pub mu: [F; D_EF],
    pub mu_pow: [F; D_EF],

    // μ PoW witness and sample for proof-of-work check
    pub mu_pow_witness: F,
    pub mu_pow_sample: F,

    // Global column index (0-indexed, increments by 1 per row within a proof
    // block, covering both valid and padding rows).
    pub global_col_idx: F,

    // Stacking claim and batched coefficient computed in OpeningClaimsCols
    pub stacking_claim: [F; D_EF],
    pub claim_coefficient: [F; D_EF],

    // Sum of each stacking_claim * claim_coefficient
    pub final_s_eval: [F; D_EF],

    // RLC of stacking claims using mu
    pub whir_claim: [F; D_EF],
}

pub struct StackingClaimsAir {
    // External buses
    pub stacking_indices_bus: StackingIndicesBus,
    pub whir_module_bus: WhirModuleBus,
    pub whir_mu_bus: WhirMuBus,
    pub transcript_bus: TranscriptBus,
    pub exp_bits_len_bus: ExpBitsLenBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub claim_coefficients_bus: ClaimCoefficientsBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,

    pub stacking_index_mult: usize,
    /// Maximum number of stacking columns per proof.
    pub w_stack: usize,
    /// Number of PoW bits for μ batching challenge.
    pub mu_pow_bits: usize,
}

impl BaseAirWithPublicValues<F> for StackingClaimsAir {}
impl PartitionedBaseAir<F> for StackingClaimsAir {}

impl<F> BaseAir<F> for StackingClaimsAir {
    fn width(&self) -> usize {
        StackingClaimsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for StackingClaimsAir
where
    AB::F: PrimeField32,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &StackingClaimsCols<AB::Var> = (*local).borrow();
        let next: &StackingClaimsCols<AB::Var> = (*next).borrow();

        let is_in_block = local.is_valid + local.is_padding;
        let next_is_in_block = next.is_valid + next.is_padding;

        NestedForLoopSubAir::<2> {}.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: is_in_block.clone(),
                    counter: [local.proof_idx.into(), local.global_col_idx.into()],
                    is_first: [local.is_first.into(), is_in_block.clone()],
                },
                NestedForLoopIoCols {
                    is_enabled: next_is_in_block.clone(),
                    counter: [next.proof_idx.into(), next.global_col_idx.into()],
                    is_first: [next.is_first.into(), next_is_in_block.clone()],
                },
            ),
        );

        // Last valid row in a proof block:
        // - valid row before padding starts, OR
        // - valid row at block end (num_valid == w_stack).
        // Degree-2 selectors to stay within max AIR degree.
        let is_last_valid = and(local.is_valid, next.is_padding + local.is_last);
        // Valid row that continues to another valid row in the same proof block:
        // excludes the terminal valid row (before padding or block end).
        let is_continuing_valid = and(
            local.is_valid,
            AB::Expr::ONE - next.is_padding - local.is_last,
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_padding);
        builder.assert_bool(local.is_last);
        // Last row in a proof block is exactly the nested-loop boundary for proof_idx.
        builder.when(is_in_block.clone()).assert_eq(
            local.is_last,
            NestedForLoopSubAir::<2>::local_is_last(
                is_in_block.clone(),
                next_is_in_block.clone(),
                next.is_first,
            ),
        );
        // Once padding starts within a proof block, it stays padding
        builder
            .when(local.is_padding * not(local.is_last))
            .assert_one(next.is_padding);
        builder.when_first_row().assert_zero(local.proof_idx);
        builder.when(local.is_first).assert_one(local.is_valid);

        /*
         * Constrain that commit_idx and stacked_col_idx increment correctly.
         */
        builder.when(local.is_first).assert_zero(local.commit_idx);
        builder
            .when(local.is_first)
            .assert_zero(local.stacked_col_idx);

        let mut when_same_proof = builder.when(is_continuing_valid.clone());
        let commit_delta = next.commit_idx - local.commit_idx;

        when_same_proof.assert_bool(commit_delta.clone());
        when_same_proof
            .when(commit_delta.clone())
            .assert_zero(next.stacked_col_idx);
        when_same_proof
            .when(not::<AB::Expr>(commit_delta))
            .assert_one(next.stacked_col_idx - local.stacked_col_idx);

        /*
         * Constrain global_col_idx: starts at 0, is forced by NestedForLoopSubAir
         * to increment by exactly 1 within each proof block, and ends at w_stack - 1.
         */
        builder
            .when(local.is_first)
            .assert_zero(local.global_col_idx);
        builder
            .when(local.is_last * is_in_block)
            .assert_eq(local.global_col_idx, AB::Expr::from_usize(self.w_stack - 1));

        self.stacking_indices_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            StackingIndexMessage {
                commit_idx: local.commit_idx,
                col_idx: local.stacked_col_idx,
            },
            local.is_valid * AB::Expr::from_usize(self.stacking_index_mult),
        );

        /*
         * Compute the running sum of stacking_claim * claim_coefficient values and then
         * constrain the final result to be equal to s_{n_stack}(u_{n_stack}), which is
         * sent from SumcheckRoundsAir.
         */
        self.claim_coefficients_bus.receive(
            builder,
            local.proof_idx,
            ClaimCoefficientsMessage {
                commit_idx: local.commit_idx,
                stacked_col_idx: local.stacked_col_idx,
                coefficient: local.claim_coefficient,
            },
            local.is_valid,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_multiply(local.stacking_claim, local.claim_coefficient),
            local.final_s_eval,
        );

        assert_array_eq(
            &mut builder.when(is_continuing_valid.clone()),
            ext_field_add(
                ext_field_multiply(next.stacking_claim, next.claim_coefficient),
                local.final_s_eval,
            ),
            next.final_s_eval,
        );

        self.sumcheck_claims_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::TWO,
                value: local.final_s_eval.map(Into::into),
            },
            is_last_valid.clone(),
        );

        /*
         * Constrain transcript operations and send the final tidx to the WHIR module.
         */
        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::TWO,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        builder
            .when(is_continuing_valid.clone())
            .assert_eq(local.tidx + AB::F::from_usize(D_EF), next.tidx);

        let mu_pow_offset = pow_tidx_count(self.mu_pow_bits);

        for i in 0..D_EF {
            // Observe stacking_claim at tidx + 0..D_EF
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i) + local.tidx,
                    value: local.stacking_claim[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            // Sample μ at tidx + D_EF + mu_pow_offset + i (after μ PoW observe/sample if any)
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i + D_EF + mu_pow_offset) + local.tidx,
                    value: local.mu[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                is_last_valid.clone(),
            );
        }

        if self.mu_pow_bits > 0 {
            // μ PoW: observe mu_pow_witness at tidx + D_EF (on last valid row only)
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(D_EF) + local.tidx,
                    value: local.mu_pow_witness.into(),
                    is_sample: AB::Expr::ZERO,
                },
                is_last_valid.clone(),
            );

            // μ PoW: sample mu_pow_sample at tidx + D_EF + 1 (on last valid row only)
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(D_EF + 1) + local.tidx,
                    value: local.mu_pow_sample.into(),
                    is_sample: AB::Expr::ONE,
                },
                is_last_valid.clone(),
            );

            // μ PoW check: g^{mu_pow_sample[0:mu_pow_bits]} = 1
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: AB::F::GENERATOR.into(),
                    bit_src: local.mu_pow_sample.into(),
                    num_bits: AB::Expr::from_usize(self.mu_pow_bits),
                    result: AB::Expr::ONE,
                },
                is_last_valid.clone(),
            );
        }

        /*
         * Compute the RLC of the stacking claims and send it to the WHIR module.
         * Running sums propagate through valid rows only (not(is_last_valid)),
         * since padding rows have zero claims and don't affect the accumulators.
         */
        assert_one_ext(&mut builder.when(local.is_first), local.mu_pow);
        assert_array_eq(
            &mut builder.when(is_continuing_valid.clone()),
            local.mu,
            next.mu,
        );
        assert_array_eq(
            &mut builder.when(is_continuing_valid.clone()),
            ext_field_multiply(local.mu, local.mu_pow),
            next.mu_pow,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.stacking_claim,
            local.whir_claim,
        );

        assert_array_eq(
            &mut builder.when(is_continuing_valid.clone()),
            ext_field_add(
                ext_field_multiply(next.stacking_claim, next.mu_pow),
                local.whir_claim,
            ),
            next.whir_claim,
        );

        // Send to WHIR module with tidx after all transcript operations
        self.whir_module_bus.send(
            builder,
            local.proof_idx,
            WhirModuleMessage {
                tidx: AB::Expr::from_usize(2 * D_EF + mu_pow_offset) + local.tidx,
                claim: local.whir_claim.map(Into::into),
            },
            is_last_valid.clone(),
        );
        self.whir_mu_bus.send(
            builder,
            local.proof_idx,
            WhirMuMessage {
                mu: local.mu.map(Into::into),
            },
            is_last_valid,
        );
    }
}
