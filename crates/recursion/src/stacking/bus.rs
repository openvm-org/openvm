use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::define_typed_per_proof_permutation_bus;

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
