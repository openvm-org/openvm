use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};
use openvm_rv32im_circuit::{BranchEqualCoreAir, BranchEqualStep};

use crate::adapters::branch_native_adapter::{BranchNativeAdapterAir, BranchNativeAdapterStep};

pub type NativeBranchEqAir = VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1>>;
pub type NativeBranchEqualStep = BranchEqualStep<BranchNativeAdapterStep, 1>;
pub type NativeBranchEqualChip<F> = NewVmChipWrapper<F, NativeBranchEqAir, NativeBranchEqualStep>;
