use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{BranchAdapterAir, U16_BITS};

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

pub type BranchLessThanAir =
    VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type BranchLessThanExecutor = BranchLessThanCoreExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type BranchLessThanChip<F> = VmChipWrapper<F, BranchLessThanFiller>;
