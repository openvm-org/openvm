use core::borrow::Borrow;

use openvm_circuit_primitives::{
    is_zero::{IsZeroAuxCols, IsZeroIo, IsZeroSubAir},
    utils::{assert_array_eq, not, or},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        BatchConstraintModuleBus, BatchConstraintModuleMessage, GkrModuleBus, GkrModuleMessage,
        TranscriptBus,
    },
    gkr::bus::{
        GkrLayerInputBus, GkrLayerInputMessage, GkrLayerOutputBus, GkrLayerOutputMessage,
        GkrXiSamplerBus, GkrXiSamplerMessage,
    },
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    subairs::proof_idx::{ProofIdxIoCols, ProofIdxSubAir},
    utils::assert_zeros,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct GkrInputCols<T> {
    /// Whether the current row is enabled (i.e. not padding)
    pub is_enabled: T,

    pub proof_idx: T,

    pub n_logup: T,
    pub n_max: T,

    /// Flag indicating whether there are any interactions
    /// n_logup = 0 <=> total_interactions = 0
    pub is_n_logup_zero: T,
    pub is_n_logup_zero_aux: IsZeroAuxCols<T>,

    pub is_n_max_greater_than_n_logup: T,

    /// Transcript index
    pub tidx: T,

    /// Root denominator claim
    pub q0_claim: [T; D_EF],

    pub alpha_logup: [T; D_EF],

    pub input_layer_claim: [[T; D_EF]; 2],

    // Grinding
    pub logup_pow_witness: T,
    pub logup_pow_sample: T,
}

/// The GkrInputAir handles reading and passing the GkrInput
pub struct GkrInputAir {
    // System Params
    pub l_skip: usize,
    pub logup_pow_bits: usize,
    // Buses
    pub gkr_module_bus: GkrModuleBus,
    pub bc_module_bus: BatchConstraintModuleBus,
    pub transcript_bus: TranscriptBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub layer_input_bus: GkrLayerInputBus,
    pub layer_output_bus: GkrLayerOutputBus,
    pub xi_sampler_bus: GkrXiSamplerBus,
}

impl<F: Field> BaseAir<F> for GkrInputAir {
    fn width(&self) -> usize {
        GkrInputCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for GkrInputAir {}
impl<F: Field> PartitionedBaseAir<F> for GkrInputAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for GkrInputAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &GkrInputCols<AB::Var> = (*local).borrow();
        let next: &GkrInputCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Proof Index Constraints
        ///////////////////////////////////////////////////////////////////////

        // This subair has the following constraints:
        // 1. Boolean enabled flag
        // 2. Disabled rows are followed by disabled rows
        // 3. Proof index increments by exactly one between enabled rows
        ProofIdxSubAir.eval(
            builder,
            (
                ProofIdxIoCols {
                    is_enabled: local.is_enabled,
                    proof_idx: local.proof_idx,
                }
                .map_into(),
                ProofIdxIoCols {
                    is_enabled: next.is_enabled,
                    proof_idx: next.proof_idx,
                }
                .map_into(),
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Base Constraints
        ///////////////////////////////////////////////////////////////////////

        // 1. Check if n_logup is zero (no logup constraints needed)
        IsZeroSubAir.eval(
            builder,
            (
                IsZeroIo::new(
                    local.n_logup.into(),
                    local.is_n_logup_zero.into(),
                    local.is_enabled.into(),
                ),
                local.is_n_logup_zero_aux.inv,
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Output Constraints
        ///////////////////////////////////////////////////////////////////////

        let has_interactions = AB::Expr::ONE - local.is_n_logup_zero;
        // Input layer claim is [0, \alpha] when no interactions
        assert_zeros(
            &mut builder.when(not::<AB::Expr>(has_interactions.clone())),
            local.input_layer_claim[0],
        );
        assert_array_eq(
            &mut builder.when(not::<AB::Expr>(has_interactions.clone())),
            local.input_layer_claim[1],
            local.alpha_logup,
        );

        ///////////////////////////////////////////////////////////////////////
        // Module Interactions
        ///////////////////////////////////////////////////////////////////////

        let num_layers = local.n_logup + AB::Expr::from_usize(self.l_skip);

        let needs_challenges = or(local.is_n_max_greater_than_n_logup, local.is_n_logup_zero);
        let num_challenges = local.n_max + AB::Expr::from_usize(self.l_skip)
            - has_interactions.clone() * num_layers.clone();

        // Add PoW and alpha, beta
        let tidx_after_pow_and_alpha_beta =
            local.tidx + AB::Expr::TWO + AB::Expr::from_usize(2 * D_EF);
        // Add GKR layers + Sumcheck
        let tidx_after_gkr_layers = tidx_after_pow_and_alpha_beta.clone()
            + has_interactions.clone()
                * num_layers.clone()
                * (num_layers.clone() + AB::Expr::TWO)
                * AB::Expr::from_usize(2 * D_EF);
        // Add separately sampled challenges
        let tidx_end = tidx_after_gkr_layers.clone()
            + needs_challenges.clone() * num_challenges.clone() * AB::Expr::from_usize(D_EF);

        // 1. GkrLayerInputBus
        // 1a. Send input to GkrLayerAir
        self.layer_input_bus.send(
            builder,
            local.proof_idx,
            GkrLayerInputMessage {
                // Skip q0_claim
                tidx: (tidx_after_pow_and_alpha_beta + AB::Expr::from_usize(D_EF))
                    * has_interactions.clone(),
                q0_claim: local.q0_claim.map(Into::into),
            },
            local.is_enabled * has_interactions.clone(),
        );
        // 2. GkrLayerOutputBus
        // 2a. Receive input layer claim from GkrLayerAir
        self.layer_output_bus.receive(
            builder,
            local.proof_idx,
            GkrLayerOutputMessage {
                tidx: tidx_after_gkr_layers.clone(),
                layer_idx_end: num_layers.clone() - AB::Expr::ONE,
                input_layer_claim: local.input_layer_claim.map(|claim| claim.map(Into::into)),
            },
            local.is_enabled * has_interactions.clone(),
        );
        // 3. GkrXiSamplerBus
        // 3a. Send input to GkrXiSamplerAir
        self.xi_sampler_bus.send(
            builder,
            local.proof_idx,
            GkrXiSamplerMessage {
                idx: has_interactions.clone() * num_layers,
                tidx: tidx_after_gkr_layers,
            },
            local.is_enabled * needs_challenges.clone(),
        );
        // 3b. Receive output from GkrXiSamplerAir
        self.xi_sampler_bus.receive(
            builder,
            local.proof_idx,
            GkrXiSamplerMessage {
                idx: local.n_max + AB::Expr::from_usize(self.l_skip - 1),
                tidx: tidx_end.clone(),
            },
            local.is_enabled * needs_challenges,
        );

        ///////////////////////////////////////////////////////////////////////
        // External Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. GkrModuleBus
        // 1a. Receive initial GKR module message on first layer
        self.gkr_module_bus.receive(
            builder,
            local.proof_idx,
            GkrModuleMessage {
                tidx: local.tidx,
                n_logup: local.n_logup,
                n_max: local.n_max,
                is_n_max_greater: local.is_n_max_greater_than_n_logup,
            },
            local.is_enabled,
        );

        // 2. TranscriptBus
        // 2a. Observe pow witness
        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            local.tidx.into(),
            local.logup_pow_witness.into(),
            local.is_enabled,
        );
        // 2b. Sample pow challenge
        self.transcript_bus.sample(
            builder,
            local.proof_idx,
            local.tidx.into() + AB::Expr::ONE,
            local.logup_pow_sample.into(),
            local.is_enabled,
        );
        // 2c. Sample alpha_logup challenge
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx.into() + AB::Expr::TWO,
            local.alpha_logup.map(Into::into),
            local.is_enabled,
        );
        // 2d. Observe `q0_claim` claim
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::TWO + AB::Expr::from_usize(2 * D_EF),
            local.q0_claim,
            local.is_enabled * has_interactions,
        );

        // 3. BatchConstraintModuleBus
        // 3a. Send input layer claims for further verification
        self.bc_module_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintModuleMessage {
                tidx: tidx_end,
                gkr_input_layer_claim: local.input_layer_claim.map(|claim| claim.map(Into::into)),
            },
            local.is_enabled,
        );

        // 4. ExpBitsLenBus
        // 4a. Check proof-of-work using `ExpBitsLenBus`.
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: AB::Expr::from_prime_subfield(
                    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield::GENERATOR,
                ),
                bit_src: local.logup_pow_sample.into(),
                num_bits: AB::Expr::from_usize(self.logup_pow_bits),
                result: AB::Expr::ONE,
            },
            local.is_enabled,
        );
    }
}
