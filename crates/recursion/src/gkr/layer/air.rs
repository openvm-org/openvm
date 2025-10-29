use core::borrow::Borrow;

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, extension::BinomiallyExtendable};
use p3_matrix::Matrix;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, XiRandomnessBus, XiRandomnessMessage},
    gkr::{
        GkrSumcheckChallengeBus, GkrSumcheckChallengeMessage,
        bus::{
            GkrLayerInputBus, GkrLayerInputMessage, GkrLayerOutputBus, GkrLayerOutputMessage,
            GkrSumcheckInputBus, GkrSumcheckInputMessage, GkrSumcheckOutputBus,
            GkrSumcheckOutputMessage,
        },
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{assert_eq_array, assert_zeros, ext_field_add, ext_field_multiply, ext_field_subtract},
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct GkrLayerCols<T> {
    pub is_enabled: T,

    pub proof_idx: T,

    pub is_first_layer: T,

    /// GKR layer index
    pub layer_idx: T,

    /// Transcript index at the start of this layer
    pub tidx: T,

    /// Sampled batching challenge
    pub lambda: [T; D_EF],

    /// Layer claims
    pub p_xi_0: [T; D_EF],
    pub q_xi_0: [T; D_EF],
    pub p_xi_1: [T; D_EF],
    pub q_xi_1: [T; D_EF],

    // (p_xi_1 - p_xi_0) * mu + p_xi_0
    pub numer_claim: [T; D_EF],
    // (q_xi_1 - q_xi_0) * mu + q_xi_0
    pub denom_claim: [T; D_EF],

    // Sumcheck claim input
    pub sumcheck_claim_in: [T; D_EF],

    /// Received from GkrLayerSumcheckAir
    pub eq_at_r_prime: [T; D_EF],

    /// Corresponds to `mu` - reduction point
    pub mu: [T; D_EF],
}

/// The GkrLayerAir handles layer-to-layer transitions in the GKR protocol
pub struct GkrLayerAir {
    // External buses
    pub xi_randomness_bus: XiRandomnessBus,
    pub transcript_bus: TranscriptBus,
    // Internal buses
    pub layer_input_bus: GkrLayerInputBus,
    pub layer_output_bus: GkrLayerOutputBus,
    pub sumcheck_input_bus: GkrSumcheckInputBus,
    pub sumcheck_output_bus: GkrSumcheckOutputBus,
    pub sumcheck_challenge_bus: GkrSumcheckChallengeBus,
}

impl<F: Field> BaseAir<F> for GkrLayerAir {
    fn width(&self) -> usize {
        GkrLayerCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for GkrLayerAir {}
impl<F: Field> PartitionedBaseAir<F> for GkrLayerAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for GkrLayerAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &GkrLayerCols<AB::Var> = (*local).borrow();
        let next: &GkrLayerCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Boolean Constraints
        ///////////////////////////////////////////////////////////////////////

        builder.assert_bool(local.is_enabled);

        ///////////////////////////////////////////////////////////////////////
        // Loop Constraints
        ///////////////////////////////////////////////////////////////////////

        type LoopSubAir = NestedForLoopSubAir<1, 0>;

        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_enabled,
                        counter: [local.proof_idx],
                        is_first: [local.is_first_layer],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_enabled,
                        counter: [next.proof_idx],
                        is_first: [next.is_first_layer],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        let is_last_layer = LoopSubAir::local_is_last(next.is_enabled, next.is_first_layer);

        // Layer index starts from 0
        builder
            .when(local.is_first_layer)
            .assert_zero(local.layer_idx);

        // Layer index increments by 1
        builder
            .when(local.is_enabled * (AB::Expr::ONE - is_last_layer.clone()))
            .assert_eq(next.layer_idx, local.layer_idx + AB::Expr::ONE);

        ///////////////////////////////////////////////////////////////////////
        // Root Layer Constraints
        ///////////////////////////////////////////////////////////////////////

        // Compute cross terms: p_cross = p_xi_0 * q_xi_1 + p_xi_1 * q_xi_0
        //                       q_cross = q_xi_0 * q_xi_1
        let (p_cross_term, q_cross_term) =
            compute_recursive_relations(local.p_xi_0, local.q_xi_0, local.p_xi_1, local.q_xi_1);

        // Zero-check: verify p_cross = 0 at root layer
        assert_zeros(
            &mut builder.when(local.is_first_layer),
            p_cross_term.clone(),
        );

        // Root consistency check: verify q_cross = q0_claim
        assert_eq_array(
            &mut builder.when(local.is_first_layer),
            q_cross_term.clone(),
            local.sumcheck_claim_in,
        );

        ///////////////////////////////////////////////////////////////////////
        // Layer Constraints
        ///////////////////////////////////////////////////////////////////////

        // Reduce to single evaluation
        let (numer_claim, denom_claim) = reduce_to_single_evaluation(
            local.p_xi_0,
            local.p_xi_1,
            local.q_xi_0,
            local.q_xi_1,
            local.mu,
        );
        assert_array_eq(builder, local.numer_claim, numer_claim);
        assert_array_eq(builder, local.denom_claim, denom_claim);

        ///////////////////////////////////////////////////////////////////////
        // Inter-Layer Constraints
        ///////////////////////////////////////////////////////////////////////

        // Next layer claim is RLC of previous layer numer_claim and denom_claim
        assert_eq_array(
            &mut builder.when(local.is_enabled * (AB::Expr::ONE - is_last_layer.clone())),
            next.sumcheck_claim_in,
            ext_field_add::<AB::Expr>(
                local.numer_claim,
                ext_field_multiply::<AB::Expr>(next.lambda, local.denom_claim),
            ),
        );

        // Transcript index increment
        let tidx_after_sumcheck = local.tidx
            // Sample lambda on non-root layer
            + (AB::Expr::ONE - local.is_first_layer) * AB::Expr::from_canonical_usize(D_EF)
            + local.layer_idx * AB::Expr::from_canonical_usize(4 * D_EF);
        let tidx_end = tidx_after_sumcheck.clone() + AB::Expr::from_canonical_usize(5 * D_EF);
        builder
            .when(local.is_enabled * (AB::Expr::ONE - is_last_layer.clone()))
            .assert_eq(next.tidx, tidx_end.clone());

        ///////////////////////////////////////////////////////////////////////
        // Module Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. GkrLayerInputBus
        // 1a. Receive GKR layers input
        self.layer_input_bus.receive(
            builder,
            local.proof_idx,
            GkrLayerInputMessage {
                tidx: local.tidx.into(),
                q0_claim: local.sumcheck_claim_in.map(Into::into),
            },
            local.is_enabled * local.is_first_layer,
        );
        // 2. GkrLayerOutputBus
        // 2a. Send GKR input layer claims back
        self.layer_output_bus.send(
            builder,
            local.proof_idx,
            GkrLayerOutputMessage {
                tidx: tidx_end,
                layer_idx_end: local.layer_idx.into(),
                input_layer_claim: [
                    local.numer_claim.map(Into::into),
                    local.denom_claim.map(Into::into),
                ],
            },
            local.is_enabled * is_last_layer.clone(),
        );
        // 3. GkrSumcheckInputBus
        // 3a. Send claim to sumcheck
        let is_non_root_layer = local.is_enabled * (AB::Expr::ONE - local.is_first_layer);
        self.sumcheck_input_bus.send(
            builder,
            local.proof_idx,
            GkrSumcheckInputMessage {
                layer_idx: local.layer_idx.into(),
                is_last_layer: is_last_layer.clone(),
                tidx: local.tidx + AB::Expr::from_canonical_usize(D_EF),
                claim: local.sumcheck_claim_in.map(Into::into),
            },
            is_non_root_layer.clone(),
        );
        // 3. GkrSumcheckOutputBus
        // 3a. Receive sumcheck results
        let sumcheck_claim_out = ext_field_multiply::<AB::Expr>(
            ext_field_add::<AB::Expr>(
                p_cross_term,
                ext_field_multiply::<AB::Expr>(local.lambda, q_cross_term),
            ),
            local.eq_at_r_prime,
        );
        self.sumcheck_output_bus.receive(
            builder,
            local.proof_idx,
            GkrSumcheckOutputMessage {
                layer_idx: local.layer_idx.into(),
                tidx: tidx_after_sumcheck.clone(),
                claim_out: sumcheck_claim_out.map(Into::into),
                eq_at_r_prime: local.eq_at_r_prime.map(Into::into),
            },
            is_non_root_layer,
        );
        // 4. GkrSumcheckChallengeBus
        // 4a. Send challenge mu
        self.sumcheck_challenge_bus.send(
            builder,
            local.proof_idx,
            GkrSumcheckChallengeMessage {
                layer_idx: local.layer_idx.into(),
                sumcheck_round: AB::Expr::ZERO,
                challenge: local.mu.map(Into::into),
            },
            local.is_enabled * (AB::Expr::ONE - is_last_layer.clone()),
        );

        ///////////////////////////////////////////////////////////////////////
        // External Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. TranscriptBus
        // 1a. Sample `lambda`
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.lambda,
            local.is_enabled * (AB::Expr::ONE - local.is_first_layer),
        );
        // 1b. Observe layer claims
        let mut tidx = tidx_after_sumcheck;
        for claim in [local.p_xi_0, local.q_xi_0, local.p_xi_1, local.q_xi_1].into_iter() {
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                tidx.clone(),
                claim,
                local.is_enabled,
            );
            tidx += AB::Expr::from_canonical_usize(D_EF);
        }
        // 1c. Sample `mu`
        self.transcript_bus
            .sample_ext(builder, local.proof_idx, tidx, local.mu, local.is_enabled);

        // 2. XiRandomnessBus
        // 2a. Send shared randomness
        self.xi_randomness_bus.send(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: AB::Expr::ZERO,
                xi: local.mu.map(Into::into),
            },
            local.is_enabled * is_last_layer,
        );
    }
}

/// Computes recursive relations from layer claims.
///
/// Returns `(p_cross_term, q_cross_term)` where:
/// - `p_cross_term = p_xi_0 * q_xi_1 + p_xi_1 * q_xi_0`
/// - `q_cross_term = q_xi_0 * q_xi_1`
fn compute_recursive_relations<F, FA>(
    p_xi_0: [F; D_EF],
    q_xi_0: [F; D_EF],
    p_xi_1: [F; D_EF],
    q_xi_1: [F; D_EF],
) -> ([FA; D_EF], [FA; D_EF])
where
    F: Into<FA> + Copy,
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D_EF>,
{
    let p_cross_term = ext_field_add::<FA>(
        ext_field_multiply::<FA>(p_xi_0, q_xi_1),
        ext_field_multiply::<FA>(p_xi_1, q_xi_0),
    );
    let q_cross_term = ext_field_multiply::<FA>(q_xi_0, q_xi_1);
    (p_cross_term, q_cross_term)
}

/// Linearly interpolates between two points at 0 and 1.
fn interpolate_linear_at_01<F, FA>(evals: [[F; D_EF]; 2], x: [F; D_EF]) -> [FA; D_EF]
where
    F: Into<FA> + Copy,
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D_EF>,
{
    let p: [FA; D_EF] = ext_field_subtract(evals[1], evals[0]);
    ext_field_add(ext_field_multiply::<FA>(p, x), evals[0])
}

/// Reduces claims to a single evaluation point using linear interpolation.
///
/// Returns `(numer, denom)` where:
/// - `numer = (p_xi_1 - p_xi_0) * mu + p_xi_0`
/// - `denom = (q_xi_1 - q_xi_0) * mu + q_xi_0`
pub(super) fn reduce_to_single_evaluation<F, FA>(
    p_xi_0: [F; D_EF],
    p_xi_1: [F; D_EF],
    q_xi_0: [F; D_EF],
    q_xi_1: [F; D_EF],
    mu: [F; D_EF],
) -> ([FA; D_EF], [FA; D_EF])
where
    F: Into<FA> + Copy,
    FA: FieldAlgebra,
    FA::F: BinomiallyExtendable<D_EF>,
{
    let numer = interpolate_linear_at_01([p_xi_0, p_xi_1], mu);
    let denom = interpolate_linear_at_01([q_xi_0, q_xi_1], mu);
    (numer, denom)
}
