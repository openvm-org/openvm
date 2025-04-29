use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{BranchEqualCoreAir, BranchEqualCoreChip};

use super::adapters::branch_native_adapter::{BranchNativeAdapterAir, BranchNativeAdapterChip};

pub type NativeBranchEqAir = VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1>>;
pub type NativeBranchEqualStep = BranchEqualStep<NativeBranchAdapterStep, 1>;
pub type NativeBranchEqualChip<F> =
    NewVmChipWrapper<F, NativeBranchEqualAir, NativeBranchEqualStep>;
