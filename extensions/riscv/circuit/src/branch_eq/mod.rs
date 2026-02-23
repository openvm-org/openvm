use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
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
    VmAirWrapper<Rv64BranchAdapterAir, BranchEqualCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv64BranchEqualExecutor =
    BranchEqualExecutor<Rv64BranchAdapterExecutor, RV32_REGISTER_NUM_LIMBS>;
pub type Rv64BranchEqualChip<F> =
    VmChipWrapper<F, BranchEqualFiller<Rv64BranchAdapterFiller, RV32_REGISTER_NUM_LIMBS>>;
