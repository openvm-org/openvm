pub mod core;

use core::NativeBranchEqualStep;

use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};
use openvm_rv32im_circuit::BranchEqualCoreAir;

use crate::adapters::branch_native_adapter::{BranchNativeAdapterAir, BranchNativeAdapterStep};

pub type NativeBranchEqAir = VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1>>;
pub type NativeBranchEqStep = NativeBranchEqualStep<BranchNativeAdapterStep>;
pub type NativeBranchEqChip<F> = NewVmChipWrapper<F, NativeBranchEqAir, NativeBranchEqStep>;
