use core::borrow::Borrow;
use std::convert::Into;

use openvm_circuit_primitives::SubAir;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, XiRandomnessBus, XiRandomnessMessage},
    gkr::bus::{GkrXiSamplerBus, GkrXiSamplerMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
};

// perf(ayush): can probably get rid of this whole air if challenges -> transcript
// interactions are constrained in batch constraint module
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct GkrXiSamplerCols<T> {
    /// Whether the current row is enabled (i.e. not padding)
    pub is_enabled: T,
    pub proof_idx: T,
    pub is_first_challenge: T,

    /// An enabled row which is not involved in any interactions
    /// but should satisfy air constraints
    pub is_dummy: T,

    /// Challenge index
    // perf(ayush): can probably remove idx if XiRandomnessMessage takes tidx instead
    pub idx: T,

    /// Sampled challenge
    pub xi: [T; D_EF],
    /// Transcript index
    pub tidx: T,
}

pub struct GkrXiSamplerAir {
    pub xi_randomness_bus: XiRandomnessBus,
    pub transcript_bus: TranscriptBus,
    pub xi_sampler_bus: GkrXiSamplerBus,
}

impl<F: Field> BaseAir<F> for GkrXiSamplerAir {
    fn width(&self) -> usize {
        GkrXiSamplerCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for GkrXiSamplerAir {}
impl<F: Field> PartitionedBaseAir<F> for GkrXiSamplerAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for GkrXiSamplerAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &GkrXiSamplerCols<AB::Var> = (*local).borrow();
        let next: &GkrXiSamplerCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Boolean Constraints
        ///////////////////////////////////////////////////////////////////////

        builder.assert_bool(local.is_dummy);

        ///////////////////////////////////////////////////////////////////////
        // Proof Index and Loop Constraints
        ///////////////////////////////////////////////////////////////////////

        type LoopSubAir = NestedForLoopSubAir<1, 0>;
        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_enabled,
                        counter: [local.proof_idx],
                        is_first: [local.is_first_challenge],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_enabled,
                        counter: [next.proof_idx],
                        is_first: [next.is_first_challenge],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols::default(),
            ),
        );

        let is_transition_challenge =
            LoopSubAir::local_is_transition(next.is_enabled, next.is_first_challenge);
        let is_last_challenge =
            LoopSubAir::local_is_last(local.is_enabled, next.is_enabled, next.is_first_challenge);

        // Challenge index increments by 1
        builder
            .when(is_transition_challenge.clone())
            .assert_eq(next.idx, local.idx + AB::Expr::ONE);

        ///////////////////////////////////////////////////////////////////////
        // Transition Constraints
        ///////////////////////////////////////////////////////////////////////

        builder
            .when(is_transition_challenge.clone())
            .assert_eq(next.tidx, local.tidx + AB::Expr::from_usize(D_EF));

        ///////////////////////////////////////////////////////////////////////
        // Module Interactions
        ///////////////////////////////////////////////////////////////////////

        let is_not_dummy = AB::Expr::ONE - local.is_dummy;

        // 1. GkrXiSamplerBus
        // 1a. Receive input from GkrInputAir
        self.xi_sampler_bus.receive(
            builder,
            local.proof_idx,
            GkrXiSamplerMessage {
                idx: local.idx.into(),
                tidx: local.tidx.into(),
            },
            local.is_first_challenge * is_not_dummy.clone(),
        );
        // 1b. Send output to GkrInputAir
        let tidx_end = local.tidx + AB::Expr::from_usize(D_EF);
        self.xi_sampler_bus.send(
            builder,
            local.proof_idx,
            GkrXiSamplerMessage {
                idx: local.idx.into(),
                tidx: tidx_end,
            },
            is_last_challenge.clone() * is_not_dummy.clone(),
        );

        ///////////////////////////////////////////////////////////////////////
        // External Interactions
        ///////////////////////////////////////////////////////////////////////

        // 1. TranscriptBus
        // 1a. Sample challenge from transcript
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.xi,
            local.is_enabled * is_not_dummy.clone(),
        );

        // 2. XiRandomnessBus
        // 2a. Send shared randomness
        self.xi_randomness_bus.send(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.idx.into(),
                xi: local.xi.map(Into::into),
            },
            local.is_enabled * is_not_dummy,
        );
    }
}
