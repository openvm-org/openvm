use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{Rv32BranchAdapterAir, Rv32BranchAdapterExecutor, Rv32BranchAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualAir =
    VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv32BranchEqualExecutor =
    BranchEqualExecutor<Rv32BranchAdapterExecutor, RV32_REGISTER_NUM_LIMBS>;
pub type Rv32BranchEqualChip<F> =
    VmChipWrapper<F, BranchEqualFiller<Rv32BranchAdapterFiller, RV32_REGISTER_NUM_LIMBS>>;
