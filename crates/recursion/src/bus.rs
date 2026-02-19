use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF};
use p3_field::PrimeCharacteristicRing;
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

            pub fn lookup_key<AB>(
                &self,
                builder: &mut AB,
                key: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                self.0.lookup_key(builder, key.to_vec(), enabled);
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
macro_rules! define_typed_per_proof_lookup_bus {
    ($Bus:ident, $Msg:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(openvm_stark_backend::interaction::LookupBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
                Self(openvm_stark_backend::interaction::LookupBus::new(bus_index))
            }

            pub fn lookup_key<AB>(
                &self,
                builder: &mut AB,
                proof_idx: impl Into<AB::Expr>,
                key: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                let key = core::iter::once(proof_idx.into())
                    .chain(key.to_vec().into_iter().map(|x| x.into()))
                    .collect::<Vec<_>>();
                self.0.lookup_key(builder, key.to_vec(), enabled);
            }

            #[inline]
            pub fn add_key_with_lookups<AB>(
                &self,
                builder: &mut AB,
                proof_idx: impl Into<AB::Expr>,
                key: $Msg<impl Into<AB::Expr> + Clone>,
                num_lookups: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                let key = core::iter::once(proof_idx.into())
                    .chain(key.to_vec().into_iter().map(|x| x.into()))
                    .collect::<Vec<_>>();
                self.0
                    .add_key_with_lookups(builder, key.to_vec(), num_lookups);
            }
        }
    };
}

#[macro_export]
macro_rules! define_typed_permutation_bus {
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
                message: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                self.0.send(builder, message.to_vec(), enabled);
            }

            #[inline]
            pub fn receive<AB>(
                &self,
                builder: &mut AB,
                message: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                self.0.receive(builder, message.to_vec(), enabled);
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
                proof_idx: impl Into<AB::Expr>,
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
                proof_idx: impl Into<AB::Expr>,
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
pub struct GkrModuleMessage<T> {
    pub tidx: T,
    pub n_logup: T,
    pub n_max: T,
    pub is_n_max_greater: T,
}

define_typed_per_proof_permutation_bus!(GkrModuleBus, GkrModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FractionFolderInputMessage<T> {
    pub num_present_airs: T,
}

define_typed_per_proof_permutation_bus!(FractionFolderInputBus, FractionFolderInputMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ExpressionClaimNMaxMessage<T> {
    pub n_max: T,
}

define_typed_per_proof_permutation_bus!(ExpressionClaimNMaxBus, ExpressionClaimNMaxMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NLiftMessage<T> {
    pub air_idx: T,
    pub n_lift: T,
}

define_typed_per_proof_permutation_bus!(NLiftBus, NLiftMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchConstraintModuleMessage<T> {
    pub tidx: T,
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

impl TranscriptBus {
    pub fn observe<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: AB::Var,
        tidx: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
        is_enabled: impl Into<AB::Expr>,
    ) {
        self.receive(
            builder,
            proof_idx,
            TranscriptBusMessage {
                tidx: tidx.into(),
                value: value.into(),
                is_sample: AB::Expr::ZERO,
            },
            is_enabled,
        )
    }

    pub fn observe_ext<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: AB::Var,
        tidx: impl Into<AB::Expr>,
        value: [impl Into<AB::Expr>; D_EF],
        is_enabled: impl Into<AB::Expr>,
    ) {
        let tidx = tidx.into();
        let is_enabled = is_enabled.into();
        for (i, x) in value.into_iter().enumerate() {
            self.receive(
                builder,
                proof_idx,
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_usize(i),
                    value: x.into(),
                    is_sample: AB::Expr::ZERO,
                },
                is_enabled.clone(),
            )
        }
    }

    pub fn observe_commit<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: AB::Var,
        tidx: impl Into<AB::Expr>,
        commit: [impl Into<AB::Expr>; DIGEST_SIZE],
        is_enabled: impl Into<AB::Expr>,
    ) {
        let tidx = tidx.into();
        let is_enabled = is_enabled.into();
        for (i, x) in commit.into_iter().enumerate() {
            self.receive(
                builder,
                proof_idx,
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_usize(i),
                    value: x.into(),
                    is_sample: AB::Expr::ZERO,
                },
                is_enabled.clone(),
            )
        }
    }

    pub fn sample<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: AB::Var,
        tidx: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
        is_enabled: impl Into<AB::Expr>,
    ) {
        self.receive(
            builder,
            proof_idx,
            TranscriptBusMessage {
                tidx: tidx.into(),
                value: value.into(),
                is_sample: AB::Expr::ONE,
            },
            is_enabled,
        )
    }

    pub fn sample_ext<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: AB::Var,
        tidx: impl Into<AB::Expr>,
        value: [impl Into<AB::Expr>; D_EF],
        is_enabled: impl Into<AB::Expr>,
    ) {
        let tidx = tidx.into();
        let is_enabled = is_enabled.into();
        for (i, x) in value.into_iter().enumerate() {
            self.receive(
                builder,
                proof_idx,
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_usize(i),
                    value: x.into(),
                    is_sample: AB::Expr::ONE,
                },
                is_enabled.clone(),
            )
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct Poseidon2PermuteMessage<T> {
    pub input: [T; POSEIDON2_WIDTH],
    pub output: [T; POSEIDON2_WIDTH],
}

define_typed_lookup_bus!(Poseidon2PermuteBus, Poseidon2PermuteMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct Poseidon2CompressMessage<T> {
    pub input: [T; POSEIDON2_WIDTH],
    pub output: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(Poseidon2CompressBus, Poseidon2CompressMessage);

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub(crate) enum AirShapeProperty {
    AirId,
    NumInteractions,
    NeedRot,
}

impl AirShapeProperty {
    pub fn to_field<T: PrimeCharacteristicRing>(self) -> T {
        T::from_u8(self as u8)
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MerkleVerifyBusMessage<T> {
    /// The idx of the merkle proof in the proof, might have additional bits (so not 0 at the root)
    /// It will be the same for all the rows in the hashing leaves part.
    pub merkle_idx: T,
    /// The total depth of the merkle proof including the leaves part, equal to merkle_proof.len()
    /// + 1 + k
    pub total_depth: T,
    /// The height of this value, [0, k) are for the hashing leaves part, [k, total_depth) are for
    /// the merkle proof part.
    pub height: T,
    /// For the leaves, it will be 0 ~ 2^k - 1, for the next intermediate values, it will be 0 ~
    /// 2^{k-1} - 1 0 for merkle proof part.
    pub leaf_sub_idx: T,
    /// Either the leaf hash, or the intermediate hash, or the sibling hash
    pub value: [T; DIGEST_SIZE],

    pub commit_major: T,
    pub commit_minor: T,
}

define_typed_per_proof_permutation_bus!(MerkleVerifyBus, MerkleVerifyBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct AirShapeBusMessage<T> {
    pub sort_idx: T,
    /// The property this message encodes.
    /// See associated enum [AirShapeProperty].
    pub property_idx: T,
    /// The value of the corresponding property.
    pub value: T,
}

define_typed_per_proof_lookup_bus!(AirShapeBus, AirShapeBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct HyperdimBusMessage<T> {
    pub sort_idx: T,
    /// Sender constrains this is `abs(log_height - l_skip)`.
    pub n_abs: T,
    /// Sender constrains this is `n < 0 ? 1 : 0`.
    pub n_sign_bit: T,
}

define_typed_per_proof_lookup_bus!(HyperdimBus, HyperdimBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct LiftedHeightsBusMessage<T> {
    pub sort_idx: T,
    pub part_idx: T,
    pub commit_idx: T,
    pub hypercube_dim: T,
    /// Sender must constraint this equals `2^log_lifted_height`.
    pub lifted_height: T,
    /// Sender must constrain this equals `max(log_height, l_skip)`.
    pub log_lifted_height: T,
}

define_typed_per_proof_lookup_bus!(LiftedHeightsBus, LiftedHeightsBusMessage);

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

define_typed_per_proof_lookup_bus!(CommitmentsBus, CommitmentsBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct CachedCommitBusMessage<T> {
    pub air_idx: T,
    pub cached_idx: T,
    pub cached_commit: [T; DIGEST_SIZE],
}

define_typed_per_proof_permutation_bus!(CachedCommitBus, CachedCommitBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct DagCommitBusMessage<T> {
    pub idx: T,
    pub values: [T; DIGEST_SIZE],
}

define_typed_per_proof_permutation_bus!(DagCommitBus, DagCommitBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct XiRandomnessMessage<T> {
    pub idx: T,
    pub xi: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(XiRandomnessBus, XiRandomnessMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct SelHypercubeBusMessage<T> {
    pub n: T,
    pub is_first: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(SelHypercubeBus, SelHypercubeBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct SelUniBusMessage<T> {
    pub n: T,
    pub is_first: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(SelUniBus, SelUniBusMessage);

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
    // pub idx: T,
    pub sort_idx: T,
    pub part_idx: T,
    pub col_idx: T,
    pub claim: [T; D_EF],
    pub is_rot: T,
}

define_typed_per_proof_permutation_bus!(ColumnClaimsBus, ColumnClaimsMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirOpeningPointMessage<T> {
    pub idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(WhirOpeningPointBus, WhirOpeningPointMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct PublicValuesBusMessage<T> {
    pub air_idx: T,
    pub pv_idx: T,
    pub value: T,
}

define_typed_per_proof_permutation_bus!(PublicValuesBus, PublicValuesBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqNegResultMessage<T> {
    // hypercube dimension where n < 0
    pub n: T,
    // 2^{l_skip + n} * eq_n(u_0, r_0)
    pub eq: [T; D_EF],
    // 2^{l_skip + n} * k_rot_n(u_0, r_0)
    pub k_rot: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqNegResultBus, EqNegResultMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqNegBaseRandMessage<T> {
    // sampled value u_0
    pub u: [T; D_EF],
    // sampled value r_0^2
    pub r_squared: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqNegBaseRandBus, EqNegBaseRandMessage);
