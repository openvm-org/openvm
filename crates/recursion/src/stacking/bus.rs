use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{define_typed_per_proof_lookup_bus, define_typed_per_proof_permutation_bus};
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
pub struct EqRandValuesLookupMessage<T> {
    pub idx: T,
    pub u: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqRandValuesLookupBus, EqRandValuesLookupMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqBaseMessage<T> {
    // eq_0(u, r)
    pub eq_u_r: [T; D_EF],
    // eq_0(u, r * omega)
    pub eq_u_r_omega: [T; D_EF],
    // eq_0(u, 1) * eq_0(r, \omega^{-1})
    pub eq_u_r_prod: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqBaseBus, EqBaseMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqKernelLookupMessage<T> {
    pub n: T,
    pub eq_in: [T; D_EF],
    pub k_rot_in: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(EqKernelLookupBus, EqKernelLookupMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqBitsLookupMessage<T> {
    // most significant num_bits bits of row_idx
    pub b_value: T,
    // n_{stack} - n_j
    pub num_bits: T,
    // eq_{n_{stack} - n_j}(u_{> n_j}, b_j)
    pub eval: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqBitsLookupBus, EqBitsLookupMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqBitsInternalMessage<T> {
    // most significant num_bits bits of row_idx without lsb
    pub b_value: T,
    // n_{stack} - n_j - 1
    pub num_bits: T,
    // eq_{n_{stack} - n_j - 1}(u_{> n_j}, b_j)
    pub eval: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqBitsInternalBus, EqBitsInternalMessage);
