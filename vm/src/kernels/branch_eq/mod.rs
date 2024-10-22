use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::adapters::jump_native_adapter::{JumpNativeAdapterAir, JumpNativeAdapterChip},
    rv32im::branch_eq::{BranchEqualCoreAir, BranchEqualCoreChip},
};

pub type KernelBranchEqAir = VmAirWrapper<JumpNativeAdapterAir<2, 0>, BranchEqualCoreAir<1>>;
pub type KernelBranchEqChip<F> =
    VmChipWrapper<F, JumpNativeAdapterChip<F, 2, 0>, BranchEqualCoreChip<1>>;
