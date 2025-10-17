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
        bus::{
            GkrSumcheckChallengeBus, GkrSumcheckChallengeMessage, GkrSumcheckInputBus,
            GkrSumcheckInputMessage, GkrSumcheckOutputBus, GkrSumcheckOutputMessage,
        },
        sub_air::{
            ExtFieldMultAuxCols, ExtFieldMultiplySubAir, InterpolateCubicAuxCols,
            InterpolateCubicSubAir,
        },
    },
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct GkrLayerSumcheckCols<T> {
    /// Is this row real (not padding)?
    pub is_real: T,

    pub proof_idx: T,

    /// GKR layer (corresponds to `round`)
    pub layer: T,

    /// Is final layer of GKR?
    pub is_final_layer: T,

    /// Transcript index
    pub tidx_beg: T,

    /// Sumcheck sub-round index within this layer (0..layer-1)
    pub sumcheck_round: T,

    /// sumcheck_round == 0?
    pub is_first_round: T,

    /// Is last round of sumcheck?
    pub is_final_round: T,

    /// s(1) in extension field
    pub ev1: [T; D_EF],
    /// s(2) in extension field
    pub ev2: [T; D_EF],
    /// s(3) in extension field
    pub ev3: [T; D_EF],

    /// The claim coming into this sub-round (either from previous sub-round or initial)
    pub claim_in: [T; D_EF],
    /// The claim going out of this sub-round (result of cubic interpolation)
    pub claim_out: [T; D_EF],

    /// Component `sumcheck_round` of the original point ξ^{(j-1)}
    /// (corresponding to `gkr_r[sumcheck_round]`)
    pub prev_challenge: [T; D_EF],
    /// The sampled challenge for this sub-round (corresponds to `ri`)
    pub challenge: [T; D_EF],

    /// The eq value coming into this sub-round
    pub eq_in: [T; D_EF],
    /// The eq value going out (updated for this round)
    pub eq_out: [T; D_EF],

    /// Auxiliary columns for cubic interpolation constraint
    pub cubic_aux: InterpolateCubicAuxCols<T>,
    /// Auxiliary columns for eq update (multiplication in extension field)
    pub eq_mult_aux: ExtFieldMultAuxCols<T>,
}

pub struct GkrLayerSumcheckAir {
    pub transcript_bus: TranscriptBus,
    pub xi_randomness_bus: XiRandomnessBus,
    pub sumcheck_input_bus: GkrSumcheckInputBus,
    pub sumcheck_output_bus: GkrSumcheckOutputBus,
    pub sumcheck_challenge_bus: GkrSumcheckChallengeBus,

    _cubic_interpolation: InterpolateCubicSubAir,
    _ext_field_multiply: ExtFieldMultiplySubAir,
}

impl GkrLayerSumcheckAir {
    pub fn new(
        transcript_bus: TranscriptBus,
        xi_randomness_bus: XiRandomnessBus,
        sumcheck_input_bus: GkrSumcheckInputBus,
        sumcheck_output_bus: GkrSumcheckOutputBus,
        sumcheck_challenge_bus: GkrSumcheckChallengeBus,
    ) -> Self {
        Self {
            transcript_bus,
            xi_randomness_bus,
            sumcheck_input_bus,
            sumcheck_output_bus,
            sumcheck_challenge_bus,
            _cubic_interpolation: InterpolateCubicSubAir::new(),
            _ext_field_multiply: ExtFieldMultiplySubAir::new(),
        }
    }
}

impl<F: Field> BaseAir<F> for GkrLayerSumcheckAir {
    fn width(&self) -> usize {
        GkrLayerSumcheckCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for GkrLayerSumcheckAir {}
impl<F: Field> PartitionedBaseAir<F> for GkrLayerSumcheckAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for GkrLayerSumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &GkrLayerSumcheckCols<AB::Var> = (*local).borrow();
        let _next: &GkrLayerSumcheckCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////

        // Boolean constraints
        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_final_layer);
        builder.assert_bool(local.is_first_round);
        builder.assert_bool(local.is_final_round);

        ///////////////////////////////////////////////////////////////////////
        // Internal Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. GkrSumcheckInputBus
        // 1a. Receive initial sumcheck input on first round
        self.sumcheck_input_bus.receive(
            builder,
            local.proof_idx,
            GkrSumcheckInputMessage {
                layer: local.layer,
                tidx: local.tidx_beg,
                claim: local.claim_in,
            },
            local.is_first_round,
        );
        // 2. GkrSumcheckOutputBus
        // 2a. Send output back to GkrLayerAir on final round
        self.sumcheck_output_bus.send(
            builder,
            local.proof_idx,
            GkrSumcheckOutputMessage {
                layer: local.layer.into(),
                tidx: local.tidx_beg.into() + AB::Expr::from_canonical_usize(4 * D_EF),
                new_claim: local.claim_out.map(Into::into),
                eq_at_r_prime: local.eq_out.map(Into::into),
            },
            local.is_final_round,
        );

        // 3. GkrSumcheckChallengeBus
        // 3a. Receive challenge from previous GKR layer sumcheck
        self.sumcheck_challenge_bus.receive(
            builder,
            local.proof_idx,
            GkrSumcheckChallengeMessage {
                layer: local.layer - AB::Expr::ONE,
                sumcheck_round: local.sumcheck_round.into(),
                challenge: local.prev_challenge.map(Into::into),
            },
            local.is_real,
        );
        // 3b. Send challenge to next GKR layer sumcheck for eq calculation
        self.sumcheck_challenge_bus.send(
            builder,
            local.proof_idx,
            GkrSumcheckChallengeMessage {
                layer: local.layer.into(),
                sumcheck_round: local.sumcheck_round.into() + AB::Expr::ONE,
                challenge: local.challenge.map(Into::into),
            },
            local.is_real * (AB::Expr::ONE - local.is_final_layer),
        );

        ///////////////////////////////////////////////////////////////////////
        // External Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. TranscriptBus
        // 1a. Observe evaluations
        let mut tidx = local.tidx_beg.into();
        for eval in [local.ev1, local.ev2, local.ev3].into_iter() {
            for (j, value) in eval.into_iter().enumerate() {
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
        // 1b. Sample challenge `ri`
        for (j, value) in local.challenge.into_iter().enumerate() {
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

        // 2. XiRandomnessBus
        // 2a. Send last challenge
        self.xi_randomness_bus.send(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.sumcheck_round + AB::Expr::ONE,
                challenge: local.challenge.map(Into::into),
            },
            local.is_final_layer,
        );
    }
}
