use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{Rv64BranchAdapterAir, U16_BITS};

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

pub type Rv64BranchLessThanAir =
    VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64BranchLessThanExecutor = BranchLessThanExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64BranchLessThanChip<F> = VmChipWrapper<F, BranchLessThanFiller>;
