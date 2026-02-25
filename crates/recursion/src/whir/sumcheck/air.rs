use core::borrow::Borrow;

use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, WhirOpeningPointBus, WhirOpeningPointMessage},
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{ext_field_multiply, interpolate_quadratic, mobius_eq_1, pow_tidx_count},
    whir::bus::{
        WhirAlphaBus, WhirAlphaMessage, WhirEqAlphaUBus, WhirEqAlphaUMessage, WhirSumcheckBus,
        WhirSumcheckBusMessage,
    },
};

/// The columns for `SumcheckAir`.
///
/// Each row in `SumcheckAir` constrains a round of sumcheck. Rows are grouped
/// into groups of size `k_whir`.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SumcheckCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub whir_round: T,
    /// A counter that goes from 0 to k_whir - 1.
    pub subidx: T,
    pub is_first_in_proof: T,
    pub is_first_in_round: T,
    /// The transcript index at the beginning of the current sumcheck round.
    pub tidx: T,
    pub ev1: [T; D_EF],
    pub ev2: [T; D_EF],
    pub folding_pow_witness: T,
    pub folding_pow_sample: T,
    pub alpha: [T; D_EF],
    pub u: [T; D_EF],
    /// The claim at the beginning of the sumcheck round.
    pub pre_claim: [T; D_EF],
    /// The claim at the end of the group of sumcheck rounds.
    pub post_group_claim: [T; D_EF],
    /// The value `eq_i(alpha, u)` on the ith sumcheck row (within a proof).
    pub eq_partial: [T; D_EF],
    /// The number of times the challenge `alpha` is looked up by other AIRs. This is
    /// unconstrained.
    pub alpha_lookup_count: T,
}

pub struct SumcheckAir {
    pub sumcheck_bus: WhirSumcheckBus,
    pub alpha_bus: WhirAlphaBus,
    pub eq_alpha_u_bus: WhirEqAlphaUBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub transcript_bus: TranscriptBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub k: usize,
    pub folding_pow_bits: usize,
    pub generator: F,
}

impl BaseAirWithPublicValues<F> for SumcheckAir {}
impl PartitionedBaseAir<F> for SumcheckAir {}

impl<F> BaseAir<F> for SumcheckAir {
    fn width(&self) -> usize {
        SumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for SumcheckAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &SumcheckCols<AB::Var> = (*local).borrow();
        let next: &SumcheckCols<AB::Var> = (*next).borrow();

        let proof_idx = local.proof_idx;
        let is_enabled = local.is_enabled;
        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        let is_same_round = next.is_enabled - next.is_first_in_round;

        NestedForLoopSubAir::<3>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled,
                    counter: [local.proof_idx, local.whir_round, local.subidx],
                    is_first: [
                        local.is_first_in_proof,
                        local.is_first_in_round,
                        local.is_enabled,
                    ],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx, next.whir_round, next.subidx],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_round,
                        next.is_enabled,
                    ],
                }
                .map_into(),
            ),
        );

        builder
            .when(local.is_enabled - is_same_round.clone())
            .assert_eq(local.subidx, AB::Expr::from_usize(self.k - 1));

        let sumcheck_idx = local.whir_round * AB::Expr::from_usize(self.k) + local.subidx;
        self.sumcheck_bus.receive(
            builder,
            proof_idx,
            WhirSumcheckBusMessage {
                tidx: local.tidx.into(),
                sumcheck_idx: sumcheck_idx.clone(),
                pre_claim: local.pre_claim.map(Into::into),
                post_claim: local.post_group_claim.map(Into::into),
            },
            local.is_first_in_round,
        );

        let post_claim = interpolate_quadratic(local.pre_claim, local.ev1, local.ev2, local.alpha);
        assert_array_eq(
            &mut builder.when(local.is_enabled - is_same_round.clone()),
            post_claim.clone(),
            local.post_group_claim,
        );

        let mut when_sumcheck_transition = builder.when(is_same_round.clone());
        when_sumcheck_transition.assert_zero(next.is_first_in_round);
        assert_array_eq(&mut when_sumcheck_transition, post_claim, next.pre_claim);
        assert_array_eq(
            &mut when_sumcheck_transition,
            next.post_group_claim,
            local.post_group_claim,
        );
        let folding_pow_offset = pow_tidx_count(self.folding_pow_bits);
        when_sumcheck_transition.assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_usize(3 * D_EF + folding_pow_offset),
        );
        // Use Möbius-adjusted equality kernel instead of eq_1 for eval-to-coeff RS encoding
        assert_array_eq(
            &mut when_sumcheck_transition,
            next.eq_partial,
            ext_field_multiply::<AB::Expr>(
                local.eq_partial,
                mobius_eq_1::<AB::Expr>(next.u, next.alpha),
            ),
        );

        assert_array_eq(
            &mut builder.when(local.is_first_in_proof),
            local.eq_partial,
            mobius_eq_1::<AB::Expr>(local.u, local.alpha),
        );

        self.transcript_bus
            .observe_ext(builder, proof_idx, local.tidx, local.ev1, is_enabled);
        self.transcript_bus.observe_ext(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_usize(D_EF),
            local.ev2,
            is_enabled,
        );
        if self.folding_pow_bits > 0 {
            self.transcript_bus.observe(
                builder,
                proof_idx,
                local.tidx + AB::Expr::from_usize(2 * D_EF),
                local.folding_pow_witness,
                is_enabled,
            );
            self.transcript_bus.sample(
                builder,
                proof_idx,
                local.tidx + AB::Expr::from_usize(2 * D_EF + 1),
                local.folding_pow_sample,
                is_enabled,
            );
        }
        self.transcript_bus.sample_ext(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_usize(2 * D_EF + folding_pow_offset),
            local.alpha,
            is_enabled,
        );

        if self.folding_pow_bits > 0 {
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: self.generator.into(),
                    bit_src: local.folding_pow_sample.into(),
                    num_bits: AB::Expr::from_usize(self.folding_pow_bits),
                    result: AB::Expr::ONE,
                },
                is_enabled,
            );
        }

        self.alpha_bus.add_key_with_lookups(
            builder,
            proof_idx,
            WhirAlphaMessage {
                idx: sumcheck_idx.clone(),
                challenge: local.alpha.map(Into::into),
            },
            local.is_enabled * local.alpha_lookup_count,
        );
        self.eq_alpha_u_bus.send(
            builder,
            proof_idx,
            WhirEqAlphaUMessage {
                value: local.eq_partial,
            },
            local.is_enabled - is_same_proof.clone(),
        );
        self.whir_opening_point_bus.receive(
            builder,
            proof_idx,
            WhirOpeningPointMessage {
                idx: sumcheck_idx,
                value: local.u.map(Into::into),
            },
            is_enabled,
        );
    }
}
