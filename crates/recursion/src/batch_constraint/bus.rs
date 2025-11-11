use p3_field::FieldAlgebra;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::define_typed_per_proof_permutation_bus;

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub(super) enum BatchConstraintInnerMessageType {
    R,
    Xi,
    Mu,
}

impl BatchConstraintInnerMessageType {
    pub fn to_field<T: FieldAlgebra>(self) -> T {
        T::from_canonical_u8(self as u8)
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchConstraintConductorMessage<T> {
    pub msg_type: T,
    pub idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
    BatchConstraintConductorBus,
    BatchConstraintConductorMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct SumcheckClaimMessage<T> {
    pub round: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(SumcheckClaimBus, SumcheckClaimMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqSharpUniMessage<T> {
    pub xi_idx: T,
    pub iter_idx: T,
    pub product: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqSharpUniBus, EqSharpUniMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqZeroNMessage<T> {
    pub is_sharp: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqZeroNBus, EqZeroNMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqNOuterMessage<T> {
    pub is_sharp: T,
    pub n: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqNOuterBus, EqNOuterMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct Eq3bMessage<T> {
    pub sort_idx: T,
    pub interaction_idx: T,
    pub eq_3b: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(Eq3bBus, Eq3bMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct SymbolicExpressionMessage<T> {
    pub air_idx: T,
    pub node_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(SymbolicExpressionBus, SymbolicExpressionMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ExpressionClaimMessage<T> {
    pub is_interaction: T,
    pub idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(ExpressionClaimBus, ExpressionClaimMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct InteractionsFoldingMessage<T> {
    pub air_idx: T,
    pub interaction_idx: T,
    pub is_mult: T,
    pub idx_in_message: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(InteractionsFoldingBus, InteractionsFoldingMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintsFoldingMessage<T> {
    pub air_idx: T,
    pub constraint_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(ConstraintsFoldingBus, ConstraintsFoldingMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqNegInternalMessage<T> {
    // hypercube dimension n * (-1)
    pub neg_n: T,
    // sampled value u_0
    pub u: [T; D_EF],
    // sampled value r_0^{2^{1 - n}}
    pub r: [T; D_EF],
    // (r_0 * omega)^{2^{1 - n}}, where omega is the D^2 generator
    pub r_omega: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqNegInternalBus, EqNegInternalMessage);
