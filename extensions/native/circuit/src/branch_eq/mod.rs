use axvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use axvm_rv32im_circuit::{BranchEqualCoreAir, BranchEqualCoreChip};

use super::adapters::branch_native_adapter::{BranchNativeAdapterAir, BranchNativeAdapterChip};

pub type KernelBranchEqAir = VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1>>;
pub type KernelBranchEqChip<F> =
    VmChipWrapper<F, BranchNativeAdapterChip<F>, BranchEqualCoreChip<1>>;
