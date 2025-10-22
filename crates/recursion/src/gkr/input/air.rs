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
    bus::{
        BatchConstraintModuleBus, BatchConstraintModuleMessage, ExpBitsLenBus, ExpBitsLenMessage,
        GkrModuleBus, GkrModuleMessage, TranscriptBus, TranscriptBusMessage,
    },
    gkr::bus::{
        GkrLayerInputBus, GkrLayerInputMessage, GkrLayerOutputBus, GkrLayerOutputMessage,
        GkrXiSamplerInputBus, GkrXiSamplerInputMessage, GkrXiSamplerOutputBus,
        GkrXiSamplerOutputMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct GkrInputCols<T> {
    /// Is this row real (not padding)?
    pub is_real: T,

    pub proof_idx: T,

    pub num_layers: T,

    /// Transcript index
    pub tidx_beg: T,
    // TODO: can this be derived?
    pub tidx_after_gkr_layers: T,
    // TODO: can this be derived?
    pub tidx_end: T,

    pub input_layer_claim: [[T; D_EF]; 2],

    // Grinding
    pub logup_pow_witness: T,
    pub logup_pow_sample: T,

    pub n_logup: T,
    pub n_max: T,
    // max(n_logup, n_max)
    pub n_global: T,

    pub is_n_logup_zero: T,
    pub is_n_logup_equal_to_n_global: T,
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
    pub xi_sampler_input_bus: GkrXiSamplerInputBus,
    pub xi_sampler_output_bus: GkrXiSamplerOutputBus,
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
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &GkrInputCols<AB::Var> = (*local).borrow();
        let _next: &GkrInputCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////

        // Boolean constraints
        builder.assert_bool(local.is_real);

        ///////////////////////////////////////////////////////////////////////
        // Internal Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. GkrLayerInputBus
        // 1a. Send input to GkrLayerAir
        self.layer_input_bus.send(
            builder,
            local.proof_idx,
            GkrLayerInputMessage {
                num_layers: local.num_layers.into(),
                // Skip indices for grinding and alpha, beta challenges
                tidx: local.tidx_beg + AB::Expr::TWO + AB::Expr::from_canonical_usize(2 * D_EF),
            },
            local.is_real * (AB::Expr::ONE - local.is_n_logup_zero),
        );
        // 2. GkrLayerOutputBus
        // 2a. Receive input layer claim from GkrLayerAir
        self.layer_output_bus.receive(
            builder,
            local.proof_idx,
            GkrLayerOutputMessage {
                tidx: local.tidx_after_gkr_layers,
                input_layer_claim: local.input_layer_claim,
            },
            local.is_real * (AB::Expr::ONE - local.is_n_logup_zero),
        );
        // 3. GkrXiSamplerInputBus
        // 3a. Send input to GkrXiSamplerAir
        self.xi_sampler_input_bus.send(
            builder,
            local.proof_idx,
            GkrXiSamplerInputMessage {
                idx_start: local.num_layers.into(),
                num_challenges: local.n_max + AB::Expr::from_canonical_usize(self.l_skip),
                tidx: local.tidx_after_gkr_layers.into(),
            },
            local.is_real * (AB::Expr::ONE - local.is_n_logup_equal_to_n_global),
        );
        // 4. GkrXiSamplerOutputBus
        // 4a. Receive input layer claim from GkrXiSamplerAir
        self.xi_sampler_output_bus.receive(
            builder,
            local.proof_idx,
            GkrXiSamplerOutputMessage {
                tidx: local.tidx_end,
            },
            local.is_real * (AB::Expr::ONE - local.is_n_logup_equal_to_n_global),
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
                tidx: local.tidx_beg.into(),
                n_logup: local.n_logup.into(),
                n_max: local.n_max.into(),
                n_global: local.n_global.into(),
            },
            local.is_real,
        );

        // 2. TranscriptBus
        // 2a. Observe pow witness
        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            TranscriptBusMessage {
                tidx: local.tidx_beg.into(),
                value: local.logup_pow_witness.into(),
                is_sample: AB::Expr::ZERO,
            },
            local.is_real,
        );
        // 2b. Sample pow challenge
        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            TranscriptBusMessage {
                tidx: local.tidx_beg.into() + AB::Expr::ONE,
                value: local.logup_pow_sample.into(),
                is_sample: AB::Expr::ONE,
            },
            local.is_real,
        );

        // 3. BatchConstraintModuleBus
        // 3a. Send input layer claims for further verification
        self.bc_module_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintModuleMessage {
                // Skip grinding nonce observation and grinding challenge sampling
                tidx_alpha_beta: local.tidx_beg.into() + AB::Expr::TWO,
                tidx: local.tidx_end.into(),
                n_max: local.n_max.into(),
                gkr_input_layer_claim: local.input_layer_claim.map(|claim| claim.map(Into::into)),
            },
            local.is_real,
        );

        // 4. ExpBitsLenBus
        // 4a. Check proof-of-work using `ExpBitsLenBus`.
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: AB::Expr::from_f(<AB::Expr as FieldAlgebra>::F::GENERATOR),
                bit_src: local.logup_pow_sample.into(),
                num_bits: AB::Expr::from_canonical_usize(self.logup_pow_bits),
                result: AB::Expr::ONE,
            },
            local.is_real,
        );
    }
}
