use core::iter;

use openvm_stark_backend::interaction::{BusIndex, InteractionBuilder, PermutationCheckBus};
use stark_backend_v2::{D_EF, DIGEST_SIZE};
use stark_recursion_circuit_derive::AlignedBorrow;

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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct RangeCheckerBusMessage<T> {
    pub value: T,
    pub max_bits: T,
}

impl<E, T: Into<E>> BusPayload<E> for RangeCheckerBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.value.into(), self.max_bits.into()]
    }
}

define_typed_bus!(RangeCheckerBus, RangeCheckerBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct PowerCheckerBusMessage<T> {
    pub log: T,
    pub exp: T,
}

impl<E, T: Into<E>> BusPayload<E> for PowerCheckerBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.log.into(), self.exp.into()]
    }
}

define_typed_bus!(PowerCheckerBus, PowerCheckerBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchConstraintModuleMessage<T> {
    pub tidx: T,
    pub n_max: T,
    pub alpha_logup: [T; 4],
    pub beta_logup: [T; 4],
    pub gkr_input_layer_claim: [[T; D_EF]; 2],
}

impl<E, T: Into<E>> BusPayload<E> for BatchConstraintModuleMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        [self.tidx.into(), self.n_max.into()]
            .into_iter()
            .chain(self.alpha_logup.map(Into::into))
            .chain(self.beta_logup.map(Into::into))
            .chain(
                self.gkr_input_layer_claim
                    .into_iter()
                    .flat_map(|claim| claim.into_iter().map(Into::into)),
            )
            .collect()
    }
}

define_typed_bus!(BatchConstraintModuleBus, BatchConstraintModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StackingModuleMessage<T> {
    pub tidx: T,
}

impl<E, T: Into<E>> BusPayload<E> for StackingModuleMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.tidx.into()]
    }
}

define_typed_bus!(StackingModuleBus, StackingModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirModuleMessage<T> {
    pub tidx: T,
}

impl<E, T: Into<E>> BusPayload<E> for WhirModuleMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![self.tidx.into()]
    }
}

define_typed_bus!(WhirModuleBus, WhirModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct AirShapeBusMessage<T> {
    pub sort_idx: T,
    pub air_id: T,
    pub hypercube_dim: T,
    pub has_preprocessed: T,
    pub num_main_parts: T,
    pub num_interactions: T,
}

impl<E, T: Into<E>> BusPayload<E> for AirShapeBusMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        vec![
            self.sort_idx.into(),
            self.air_id.into(),
            self.hypercube_dim.into(),
            self.has_preprocessed.into(),
            self.num_main_parts.into(),
            self.num_interactions.into(),
        ]
    }
}

define_typed_bus!(AirShapeBus, AirShapeBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct XiRandomnessMessage<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

impl<E, T: Into<E>> BusPayload<E> for XiRandomnessMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.idx.into())
            .chain(self.challenge.map(Into::into))
            .collect()
    }
}

define_typed_bus!(XiRandomnessBus, XiRandomnessMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ColumnClaimsMessage<T> {
    pub idx: T,
    pub sort_idx: T,
    pub part_idx: T,
    pub col_idx: T,
    pub col_claim: [T; D_EF],
    pub rot_claim: [T; D_EF],
}

impl<E, T: Into<E>> BusPayload<E> for ColumnClaimsMessage<T> {
    fn into_bus_vec(self) -> Vec<E> {
        iter::once(self.idx.into())
            .chain(iter::once(self.sort_idx.into()))
            .chain(iter::once(self.part_idx.into()))
            .chain(iter::once(self.col_idx.into()))
            .chain(self.col_claim.map(Into::into))
            .chain(self.rot_claim.map(Into::into))
            .collect()
    }
}

define_typed_bus!(ColumnClaimsBus, ColumnClaimsMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
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
