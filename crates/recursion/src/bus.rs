use core::iter;

use openvm_stark_backend::interaction::{BusIndex, InteractionBuilder, PermutationCheckBus};
use stark_backend_v2::{D_EF, DIGEST_SIZE};

pub trait BusPayload<E> {
    fn into_bus_vec(self) -> Vec<E>;
}

macro_rules! define_typed_bus {
    ($Bus:ident, $Msg:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(PermutationCheckBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: BusIndex) -> Self {
                Self(PermutationCheckBus::new(bus_index))
            }

            #[inline]
            pub fn send<AB>(
                &self,
                builder: &mut AB,
                message: $Msg<impl Into<AB::Expr>>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: InteractionBuilder,
                $Msg<AB::Expr>: BusPayload<AB::Expr>,
            {
                self.0.send(builder, message.into_bus_vec(), enabled);
            }

            #[inline]
            pub fn receive<AB>(
                &self,
                builder: &mut AB,
                message: $Msg<impl Into<AB::Expr>>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: InteractionBuilder,
                $Msg<AB::Expr>: BusPayload<AB::Expr>,
            {
                self.0.receive(builder, message.into_bus_vec(), enabled);
            }
        }
    };
}

pub struct GkrModuleMessage<T> {
    pub tidx: T,
    pub n_logup: T,
    pub n_max: T,
}

impl<E, T: Into<E>> BusPayload<E> for GkrModuleMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.tidx.into(), self.n_logup.into(), self.n_max.into()]
    }
}

define_typed_bus!(GkrModuleBus, GkrModuleMessage);

pub struct BatchConstraintModuleMessage<T> {
    pub tidx: T,
    pub alpha_beta_tidx: T,
    pub n_max: T,
    pub gkr_input_layer_claim: [[T; D_EF]; 2],
}

impl<E, T: Into<E>> BusPayload<E> for BatchConstraintModuleMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        [
            self.tidx.into(),
            self.alpha_beta_tidx.into(),
            self.n_max.into(),
        ]
        .into_iter()
        .chain(
            self.gkr_input_layer_claim
                .into_iter()
                .flat_map(|claim| claim.into_iter().map(Into::into)),
        )
        .collect()
    }
}

define_typed_bus!(BatchConstraintModuleBus, BatchConstraintModuleMessage);

pub struct StackingModuleMessage<T> {
    pub tidx: T,
}

impl<E, T: Into<E>> BusPayload<E> for StackingModuleMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.tidx.into()]
    }
}

define_typed_bus!(StackingModuleBus, StackingModuleMessage);

pub struct WhirModuleMessage<T> {
    pub tidx: T,
}

impl<E, T: Into<E>> BusPayload<E> for WhirModuleMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.tidx.into()]
    }
}

define_typed_bus!(WhirModuleBus, WhirModuleMessage);

#[derive(Debug)]
pub struct TranscriptBusMessage<T> {
    pub tidx: T,
    pub value: T,
    pub is_sample: T,
}

impl<E, T: Into<E>> BusPayload<E> for TranscriptBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.tidx.into(), self.value.into(), self.is_sample.into()]
    }
}

define_typed_bus!(TranscriptBus, TranscriptBusMessage);

#[derive(Debug)]
pub struct AirShapeBusMessage<T> {
    pub sort_idx: T,
    pub idx: T,
    pub hypercube_dim: T,
    pub has_preprocessed: T,
    pub num_main_parts: T,
    pub num_interactions: T,
}

impl<E, T: Into<E>> BusPayload<E> for AirShapeBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![
            self.sort_idx.into(),
            self.idx.into(),
            self.hypercube_dim.into(),
            self.has_preprocessed.into(),
            self.num_main_parts.into(),
            self.num_interactions.into(),
        ]
    }
}

define_typed_bus!(AirShapeBus, AirShapeBusMessage);

#[derive(Debug)]
pub struct AirPartShapeBusMessage<T> {
    pub idx: T,
    pub part: T,
    pub width: T,
}

impl<E, T: Into<E>> BusPayload<E> for AirPartShapeBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.idx.into(), self.part.into(), self.width.into()]
    }
}

define_typed_bus!(AirPartShapeBus, AirPartShapeBusMessage);

#[derive(Debug)]
pub struct StackingWidthBusMessage<T> {
    pub commit_idx: T,
    pub width: T,
}

impl<E, T: Into<E>> BusPayload<E> for StackingWidthBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.commit_idx.into(), self.width.into()]
    }
}

define_typed_bus!(StackingWidthsBus, StackingWidthBusMessage);

#[derive(Debug)]
pub struct StackingCommitmentsBusMessage<T> {
    pub commit_idx: T,
    pub commitment: [T; DIGEST_SIZE],
}

impl<E, T: Into<E>> BusPayload<E> for StackingCommitmentsBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.commit_idx.into())
            .chain(self.commitment.map(Into::into))
            .collect()
    }
}

define_typed_bus!(StackingCommitmentsBus, StackingCommitmentsBusMessage);

pub struct GkrRandomnessMessage<T> {
    pub idx: T,
    pub layer: T,
    pub challenge: [T; D_EF],
}

impl<E, T: Into<E>> BusPayload<E> for GkrRandomnessMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.idx.into())
            .chain(iter::once(self.layer.into()))
            .chain(self.challenge.map(Into::into))
            .collect()
    }
}

define_typed_bus!(GkrRandomnessBus, GkrRandomnessMessage);

pub struct InitialZerocheckRandomnessMessage<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

impl<E, T: Into<E>> BusPayload<E> for InitialZerocheckRandomnessMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.idx.into())
            .chain(self.challenge.map(Into::into))
            .collect()
    }
}

define_typed_bus!(
    InitialZerocheckRandomnessBus,
    InitialZerocheckRandomnessMessage
);

pub struct ConstraintSumcheckRandomness<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

impl<E, T: Into<E>> BusPayload<E> for ConstraintSumcheckRandomness<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.idx.into())
            .chain(self.challenge.map(Into::into))
            .collect()
    }
}

define_typed_bus!(
    ConstraintSumcheckRandomnessBus,
    ConstraintSumcheckRandomness
);

pub struct ColumnClaimsMessage<T> {
    pub idx: T,
    pub air_idx: T,
    pub part_idx: T,
    pub col_idx: T,
    pub col_claim: [T; D_EF],
    pub rot_claim: [T; D_EF],
}

impl<E, T: Into<E>> BusPayload<E> for ColumnClaimsMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.idx.into())
            .chain(iter::once(self.air_idx.into()))
            .chain(iter::once(self.part_idx.into()))
            .chain(iter::once(self.col_idx.into()))
            .chain(self.col_claim.map(Into::into))
            .chain(self.rot_claim.map(Into::into))
            .collect()
    }
}

define_typed_bus!(ColumnClaimsBus, ColumnClaimsMessage);

pub struct StackingClaimsMessage<T> {
    pub idx: T,
    pub claim: [T; D_EF],
}

impl<E, T: Into<E>> BusPayload<E> for StackingClaimsMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.idx.into())
            .chain(self.claim.map(Into::into))
            .collect()
    }
}

define_typed_bus!(StackingClaimsBus, StackingClaimsMessage);

pub struct StackingSumcheckRandomnessMessage<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

impl<E, T: Into<E>> BusPayload<E> for StackingSumcheckRandomnessMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.idx.into())
            .chain(self.challenge.map(Into::into))
            .collect()
    }
}

define_typed_bus!(
    StackingSumcheckRandomnessBus,
    StackingSumcheckRandomnessMessage
);

#[derive(Debug)]
pub struct PublicValuesBusMessage<T> {
    pub air_idx: T,
    pub pv_idx: T,
    pub value: T,
}

impl<E, T: Into<E>> BusPayload<E> for PublicValuesBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.air_idx.into(), self.pv_idx.into(), self.value.into()]
    }
}

define_typed_bus!(PublicValuesBus, PublicValuesBusMessage);
