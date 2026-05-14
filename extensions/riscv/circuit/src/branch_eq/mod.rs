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

/// Pattern B: each RV64 register read returns `BLOCK_FE_WIDTH` u16 cells and the core compares
/// limb-by-limb with `LIMB_BITS = 16` range checks.
pub const RV64_BRANCH_NUM_LIMBS: usize = BLOCK_FE_WIDTH;
pub const RV64_BRANCH_LIMB_BITS: usize = 16;

pub type Rv64BranchEqualAir = VmAirWrapper<
    Rv64BranchAdapterAir,
    BranchEqualCoreAir<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>,
>;
pub type Rv64BranchEqualExecutor =
    BranchEqualExecutor<Rv64BranchAdapterExecutor, RV64_BRANCH_NUM_LIMBS>;
pub type Rv64BranchEqualChip<F> = VmChipWrapper<
    F,
    BranchEqualFiller<Rv64BranchAdapterFiller, RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>,
>;
