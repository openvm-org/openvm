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

define_typed_per_proof_permutation_bus!(EqKernelLookupBus, EqKernelLookupMessage);

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
