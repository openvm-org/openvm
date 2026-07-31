use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::Rv64BranchAdapterAir;

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

pub type Rv64BranchEqualAir =
    VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<BLOCK_FE_WIDTH>>;
pub type Rv64BranchEqualExecutor = BranchEqualExecutor<BLOCK_FE_WIDTH>;
pub type Rv64BranchEqualChip<F> = VmChipWrapper<F, BranchEqualFiller>;
