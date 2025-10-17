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
    gkr::bus::{
        GkrXiSamplerInputBus, GkrXiSamplerInputMessage, GkrXiSamplerOutputBus,
        GkrXiSamplerOutputMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct GkrXiSamplerCols<T> {
    /// Is this row real (not padding)?
    pub is_real: T,

    pub proof_idx: T,

    pub is_first_row: T,
    // xi_index == num_challenges - 1?
    pub is_final_row: T,

    /// Transcript index
    pub tidx: T,

    pub xi_index: T,
    pub num_challenges: T,

    // Sampled challenge
    pub challenge: [T; D_EF],
}

pub struct GkrXiSamplerAir {
    pub xi_randomness_bus: XiRandomnessBus,
    pub transcript_bus: TranscriptBus,
    pub xi_sampler_input_bus: GkrXiSamplerInputBus,
    pub xi_sampler_output_bus: GkrXiSamplerOutputBus,
}

impl<F: Field> BaseAir<F> for GkrXiSamplerAir {
    fn width(&self) -> usize {
        GkrXiSamplerCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for GkrXiSamplerAir {}
impl<F: Field> PartitionedBaseAir<F> for GkrXiSamplerAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for GkrXiSamplerAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &GkrXiSamplerCols<AB::Var> = (*local).borrow();
        let _next: &GkrXiSamplerCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////

        // Boolean constraints
        builder.assert_bool(local.is_real);

        ///////////////////////////////////////////////////////////////////////
        // Internal Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. GkrXiSamplerInputBus
        // 1a. Receive input from GkrInputAir
        self.xi_sampler_input_bus.receive(
            builder,
            local.proof_idx,
            GkrXiSamplerInputMessage {
                idx_start: local.xi_index,
                num_challenges: local.num_challenges,
                tidx: local.tidx,
            },
            local.is_first_row,
        );
        // 2. GkrXiSamplerOutputBus
        // 2a. Send output to GkrInputAir
        self.xi_sampler_output_bus.send(
            builder,
            local.proof_idx,
            GkrXiSamplerOutputMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(D_EF),
            },
            local.is_final_row,
        );

        ///////////////////////////////////////////////////////////////////////
        // External Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. TranscriptBus
        // 1a. Send transcript message
        for (j, value) in local.challenge.into_iter().enumerate() {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_canonical_usize(j),
                    value: value.into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_real,
            );
        }

        // 2. XiRandomnessBus
        // 2a. Send shared randomness
        self.xi_randomness_bus.send(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.xi_index.into(),
                challenge: local.challenge.map(Into::into),
            },
            local.is_real,
        );
    }
}
