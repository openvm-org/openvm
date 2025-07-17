use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::BranchEqualCoreAir;

mod core;
pub use core::*;

use crate::adapters::{BranchNativeAdapterAir, BranchNativeAdapterFiller, BranchNativeAdapterStep};

#[cfg(test)]
mod tests;

pub type NativeBranchEqAir = VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1>>;
pub type NativeBranchEqStep = NativeBranchEqualStep<BranchNativeAdapterStep>;
pub type NativeBranchEqChip<F> =
    VmChipWrapper<F, NativeBranchEqualFiller<BranchNativeAdapterFiller>>;
