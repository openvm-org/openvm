use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::adapters::branch_native_adapter::{BranchNativeAdapterAir, BranchNativeAdapterChip},
    rv32im::{BranchEqualCoreAir, BranchEqualCoreChip},
};

pub type KernelBranchEqAir = VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1>>;
pub type KernelBranchEqChip<F> =
    VmChipWrapper<F, BranchNativeAdapterChip<F>, BranchEqualCoreChip<1>>;
