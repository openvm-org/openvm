use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::adapters::native_adapter::{GenericNativeAdapterAir, GenericNativeAdapterChip},
    rv32im::branch_eq::{BranchEqualCoreAir, BranchEqualCoreChip},
};

pub type KernelBranchEqAir = VmAirWrapper<GenericNativeAdapterAir<2, 0>, BranchEqualCoreAir<1>>;
pub type KernelBranchEqChip<F> =
    VmChipWrapper<F, GenericNativeAdapterChip<F, 2, 0>, BranchEqualCoreChip<1>>;
