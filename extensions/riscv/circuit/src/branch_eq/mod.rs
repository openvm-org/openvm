use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::BranchAdapterAir;

mod core;
mod execution;
pub(crate) mod trace;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type BranchEqualAir = VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<BLOCK_FE_WIDTH>>;
pub type BranchEqualExecutor = BranchEqualCoreExecutor<BLOCK_FE_WIDTH>;
pub type BranchEqualChip<F> = VmChipWrapper<F, BranchEqualFiller>;
