use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64BranchAdapterAir, Rv64BranchAdapterExecutor, Rv64BranchAdapterFiller},
    branch_eq::{RV64_BRANCH_LIMB_BITS, RV64_BRANCH_NUM_LIMBS},
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

pub type Rv64BranchLessThanAir = VmAirWrapper<
    Rv64BranchAdapterAir,
    BranchLessThanCoreAir<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>,
>;
pub type Rv64BranchLessThanExecutor = BranchLessThanExecutor<
    Rv64BranchAdapterExecutor,
    RV64_BRANCH_NUM_LIMBS,
    RV64_BRANCH_LIMB_BITS,
>;
pub type Rv64BranchLessThanChip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<Rv64BranchAdapterFiller, RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>,
>;
