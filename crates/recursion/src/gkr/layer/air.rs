use core::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::Matrix;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, TranscriptBusMessage, XiRandomnessBus, XiRandomnessMessage},
    gkr::{
        GkrSumcheckChallengeBus, GkrSumcheckChallengeMessage,
        bus::{
            GkrLayerInputBus, GkrLayerInputMessage, GkrLayerOutputBus, GkrLayerOutputMessage,
            GkrSumcheckInputBus, GkrSumcheckInputMessage, GkrSumcheckOutputBus,
            GkrSumcheckOutputMessage,
        },
    },
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct GkrLayerCols<T> {
    /// Is this row real (not padding)?
    pub is_real: T,

    pub proof_idx: T,

    /// Transcript index at the start of this layer
    pub tidx_beg: T,
    // TODO: can this be derived?
    pub tidx_after_sumcheck: T,

    /// GKR layer columns
    pub num_layers: T,
    pub layer: T,
    pub is_first_layer: T,
    pub is_final_layer: T,

    /// Sampled batching challenge
    pub lambda: [T; D_EF],

    /// Root denominator claim
    pub q0_claim: [T; D_EF],

    /// p_xi_0
    pub numer0: [T; D_EF],
    /// q_xi_0
    pub denom0: [T; D_EF],
    /// p_xi_1
    pub numer1: [T; D_EF],
    /// q_xi_1
    pub denom1: [T; D_EF],

    /// p_xi_0 + (p_xi_1 - p_xi_0) * mu
    pub numer_claim: [T; D_EF],
    /// q_xi_0 + (q_xi_1 - q_xi_0) * mu
    pub denom_claim: [T; D_EF],

    /// numer_claim + lambda * denom_claim
    pub claim: [T; D_EF],

    /// Received from GkrLayerSumcheckAir
    pub new_claim: [T; D_EF],

    /// p_xi_0 * q_xi_1 + p_xi_1 * q_xi_0
    pub p_cross_term: [T; D_EF],
    /// q_xi_1 * q_xi_0
    pub q_cross_term: [T; D_EF],

    /// Received from GkrLayerSumcheckAir
    pub eq_at_r_prime: [T; D_EF],

    /// (p_cross_term + lambda * q_cross_term) * eq_at_r_prime
    pub expected_claim: [T; D_EF],

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

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for GkrLayerAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &GkrLayerCols<AB::Var> = (*local).borrow();
        let _next: &GkrLayerCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////

        // Boolean constraints
        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_first_layer);
        builder.assert_bool(local.is_final_layer);

        ///////////////////////////////////////////////////////////////////////
        // Internal Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. GkrLayerInputBus
        // 1a. Receive GKR layers input
        self.layer_input_bus.receive(
            builder,
            local.proof_idx,
            GkrLayerInputMessage {
                num_layers: local.num_layers.into(),
                tidx: local.tidx_beg.into(),
            },
            local.is_first_layer,
        );
        // 2. GkrLayerOutputBus
        // 2a. Send GKR input layer claims back
        self.layer_output_bus.send(
            builder,
            local.proof_idx,
            GkrLayerOutputMessage {
                tidx: local.tidx_after_sumcheck.into() + AB::Expr::from_canonical_usize(5 * D_EF),
                input_layer_claim: [
                    local.numer_claim.map(Into::into),
                    local.denom_claim.map(Into::into),
                ],
            },
            local.is_final_layer,
        );
        // 3. GkrSumcheckInputBus
        // 3a. Send claim to sumcheck
        let is_non_root_layer = local.is_real * (AB::Expr::ONE - local.is_first_layer);
        self.sumcheck_input_bus.send(
            builder,
            local.proof_idx,
            GkrSumcheckInputMessage {
                layer: local.layer.into(),
                tidx: local.tidx_beg + AB::Expr::from_canonical_usize(D_EF),
                claim: local.claim.map(Into::into),
            },
            is_non_root_layer.clone(),
        );
        // 3. GkrSumcheckOutputBus
        // 3a. Receive sumcheck results
        self.sumcheck_output_bus.receive(
            builder,
            local.proof_idx,
            GkrSumcheckOutputMessage {
                layer: local.layer,
                tidx: local.tidx_after_sumcheck,
                new_claim: local.new_claim,
                eq_at_r_prime: local.eq_at_r_prime,
            },
            is_non_root_layer,
        );
        // 4. GkrSumcheckChallengeBus
        // 4a. Send challenge mu
        self.sumcheck_challenge_bus.send(
            builder,
            local.proof_idx,
            GkrSumcheckChallengeMessage {
                layer: local.layer.into(),
                sumcheck_round: AB::Expr::ZERO,
                challenge: local.mu.map(Into::into),
            },
            local.is_real * (AB::Expr::ONE - local.is_final_layer),
        );

        ///////////////////////////////////////////////////////////////////////
        // External Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. TranscriptBus
        // 1a. Observe `q0_claim` claim
        // TODO: Create helper function for observing/sample ext element
        let mut tidx = local.tidx_beg.into();
        for (j, value) in local.q0_claim.into_iter().enumerate() {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_canonical_usize(j),
                    value: value.into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_first_layer,
            );
        }
        // 1b. Sample `lambda`
        for (j, value) in local.lambda.into_iter().enumerate() {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_canonical_usize(j),
                    value: value.into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_real * (AB::Expr::ONE - local.is_first_layer),
            );
        }
        tidx += AB::Expr::from_canonical_usize(D_EF);
        // TODO: No magic number 4, make number of sumcheck transcript interactions a constant and
        // debug assert the value in sumcheck air
        tidx += local.layer * AB::Expr::from_canonical_usize(4 * D_EF);
        // 1c. Observe layer claims
        for claim in [local.numer0, local.denom0, local.numer1, local.denom1].into_iter() {
            for (j, value) in claim.into_iter().enumerate() {
                self.transcript_bus.receive(
                    builder,
                    local.proof_idx,
                    TranscriptBusMessage {
                        tidx: tidx.clone() + AB::Expr::from_canonical_usize(j),
                        value: value.into(),
                        is_sample: AB::Expr::ZERO,
                    },
                    local.is_real,
                );
            }
            tidx += AB::Expr::from_canonical_usize(D_EF);
        }
        // 1d. Sample `mu`
        for (j, value) in local.mu.into_iter().enumerate() {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_canonical_usize(j),
                    value: value.into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_real,
            );
        }
        tidx += AB::Expr::from_canonical_usize(D_EF);

        // 2. XiRandomnessBus
        // 2a. Send shared randomness
        self.xi_randomness_bus.send(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: AB::Expr::ZERO,
                challenge: local.mu.map(Into::into),
            },
            local.is_final_layer,
        );
    }
}
