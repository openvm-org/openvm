use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv64BranchAdapterAir, Rv64BranchAdapterExecutor, Rv64BranchAdapterFiller};

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
    BranchLessThanCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>,
>;
pub type Rv64BranchLessThanExecutor =
    BranchLessThanExecutor<Rv64BranchAdapterExecutor, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>;
pub type Rv64BranchLessThanChip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<Rv64BranchAdapterFiller, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>,
>;
