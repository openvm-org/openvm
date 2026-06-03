use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{Rv64BranchAdapterAir, Rv64BranchAdapterExecutor, Rv64BranchAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64BranchEqualAir =
    VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<BLOCK_FE_WIDTH>>;
pub type Rv64BranchEqualExecutor = BranchEqualExecutor<Rv64BranchAdapterExecutor, BLOCK_FE_WIDTH>;
pub type Rv64BranchEqualChip<F> =
    VmChipWrapper<F, BranchEqualFiller<Rv64BranchAdapterFiller, BLOCK_FE_WIDTH>>;
