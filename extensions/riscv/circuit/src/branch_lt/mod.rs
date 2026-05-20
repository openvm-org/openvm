use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BranchAdapterAir, Rv64BranchAdapterExecutor, Rv64BranchAdapterFiller, RV64_U16_LIMB_BITS,
};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "aot")]
mod aot;

#[cfg(test)]
mod tests;

pub type Rv64BranchLessThanAir =
    VmAirWrapper<Rv64BranchAdapterAir, BranchLessThanCoreAir<BLOCK_FE_WIDTH, RV64_U16_LIMB_BITS>>;
pub type Rv64BranchLessThanExecutor =
    BranchLessThanExecutor<Rv64BranchAdapterExecutor, BLOCK_FE_WIDTH, RV64_U16_LIMB_BITS>;
pub type Rv64BranchLessThanChip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<Rv64BranchAdapterFiller, BLOCK_FE_WIDTH, RV64_U16_LIMB_BITS>,
>;
