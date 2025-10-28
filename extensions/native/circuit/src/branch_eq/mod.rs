use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::BranchEqualCoreAir;

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

use crate::adapters::{
    BranchNativeAdapterAir, BranchNativeAdapterExecutor, BranchNativeAdapterFiller,
};

#[cfg(test)]
mod tests;

pub type NativeBranchEqAir = VmAirWrapper<BranchNativeAdapterAir, BranchEqualCoreAir<1>>;
pub type NativeBranchEqExecutor = NativeBranchEqualExecutor<BranchNativeAdapterExecutor>;
pub type NativeBranchEqChip<F> =
    VmChipWrapper<F, NativeBranchEqualFiller<BranchNativeAdapterFiller>>;
