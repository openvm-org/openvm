use stark_backend_v2::{D_EF, DIGEST_SIZE};
use stark_recursion_circuit_derive::AlignedBorrow;

#[macro_export]
macro_rules! define_typed_lookup_bus {
    ($Bus:ident, $Msg:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(openvm_stark_backend::interaction::LookupBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
                Self(openvm_stark_backend::interaction::LookupBus::new(bus_index))
            }

            #[inline]
            pub fn add_key_with_lookups<AB>(
                &self,
                builder: &mut AB,
                key: $Msg<impl Into<AB::Expr> + Clone>,
                num_lookups: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                self.0
                    .add_key_with_lookups(builder, key.to_vec(), num_lookups);
            }
        }
    };
}

#[macro_export]
macro_rules! define_typed_per_proof_permutation_bus {
    ($Bus:ident, $Msg:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(openvm_stark_backend::interaction::PermutationCheckBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
                Self(openvm_stark_backend::interaction::PermutationCheckBus::new(
                    bus_index,
                ))
            }

            #[inline]
            pub fn send<AB>(
                &self,
                builder: &mut AB,
                proof_idx: AB::Var,
                message: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                let message = core::iter::once(proof_idx.into())
                    .chain(message.to_vec().into_iter().map(|x| x.into()))
                    .collect::<Vec<_>>();
                self.0.send(builder, message, enabled);
            }

            #[inline]
            pub fn receive<AB>(
                &self,
                builder: &mut AB,
                proof_idx: AB::Var,
                message: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                let message = core::iter::once(proof_idx.into())
                    .chain(message.to_vec().into_iter().map(|x| x.into()))
                    .collect::<Vec<_>>();
                self.0.receive(builder, message, enabled);
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

define_typed_lookup_bus!(RangeCheckerBus, RangeCheckerBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct PowerCheckerBusMessage<T> {
    pub log: T,
    pub exp: T,
}

define_typed_lookup_bus!(PowerCheckerBus, PowerCheckerBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrModuleMessage<T> {
    pub tidx: T,
    pub n_logup: T,
    pub n_max: T,
}

define_typed_per_proof_permutation_bus!(GkrModuleBus, GkrModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchConstraintModuleMessage<T> {
    pub tidx: T,
    pub n_max: T,
    pub alpha_logup: [T; 4],
    pub beta_logup: [T; 4],
    pub gkr_input_layer_claim: [[T; D_EF]; 2],
}

define_typed_per_proof_permutation_bus!(BatchConstraintModuleBus, BatchConstraintModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StackingModuleMessage<T> {
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(StackingModuleBus, StackingModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirModuleMessage<T> {
    /// The `tidx` _after_ batching randomness `mu` is sampled.
    pub tidx: T,
    /// The batching randomness to combine stacking claims.
    pub mu: [T; 4],
    /// The reduced opening claim after batching.
    pub claim: [T; 4],
}

define_typed_per_proof_permutation_bus!(WhirModuleBus, WhirModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct TranscriptBusMessage<T> {
    pub tidx: T,
    pub value: T,
    pub is_sample: T,
}

define_typed_per_proof_permutation_bus!(TranscriptBus, TranscriptBusMessage);

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

define_typed_per_proof_permutation_bus!(AirShapeBus, AirShapeBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct AirPartShapeBusMessage<T> {
    pub idx: T,
    pub part: T,
    pub width: T,
}

define_typed_per_proof_permutation_bus!(AirPartShapeBus, AirPartShapeBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StackingIndexMessage<T> {
    pub commit_idx: T,
    pub col_idx: T,
}

define_typed_per_proof_permutation_bus!(StackingIndicesBus, StackingIndexMessage);

/// Carries all commitments in the proof.
///
/// The stacking commitments have `major_idx = 0` and `minor_idx =
/// stacking_matrix_idx`. The WHIR commitments have `major_idx = whir_round + 1`
/// and `minor_idx = 0`.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct CommitmentsBusMessage<T> {
    pub major_idx: T,
    pub minor_idx: T,
    pub commitment: [T; DIGEST_SIZE],
}

define_typed_per_proof_permutation_bus!(CommitmentsBus, CommitmentsBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct XiRandomnessMessage<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(XiRandomnessBus, XiRandomnessMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintSumcheckRandomness<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
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

define_typed_per_proof_permutation_bus!(ColumnClaimsBus, ColumnClaimsMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StackingSumcheckRandomnessMessage<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
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

define_typed_per_proof_permutation_bus!(PublicValuesBus, PublicValuesBusMessage);

/*
 * STACKING MODULE INTERNAL BUSES
 */
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StackingModuleTidxMessage<T> {
    pub module_idx: T,
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(StackingModuleTidxBus, StackingModuleTidxMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ClaimCoefficientsMessage<T> {
    pub commit_idx: T,
    pub stacked_col_idx: T,
    pub coefficient: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(ClaimCoefficientsBus, ClaimCoefficientsMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct SumcheckClaimsMessage<T> {
    pub module_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(SumcheckClaimsBus, SumcheckClaimsMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqBaseMessage<T> {
    // eq_0(u, r)
    pub eq_u_r: T,
    // eq_0(u, 1) * eq_0(r, \omega^{-1})
    pub eq_u_r_prod: T,
}

define_typed_per_proof_permutation_bus!(EqBaseBus, EqBaseMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqKernelLookupMessage<T> {
    pub is_rot: T,
    pub n_j: T,
    pub value: T,
}

define_typed_per_proof_permutation_bus!(EqKernelLookupBus, EqKernelLookupMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqBitsLookupMessage<T> {
    pub n_j: T,
    pub row_idx: T,
    // eq_{n_{stack} - n_j}(u_{> n_j}, b_j)
    pub value: T,
}

define_typed_per_proof_permutation_bus!(EqBitsLookupBus, EqBitsLookupMessage);
