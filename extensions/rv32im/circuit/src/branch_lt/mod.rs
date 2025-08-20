use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
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

pub type Rv32BranchLessThanAir = VmAirWrapper<
    Rv32BranchAdapterAir,
    BranchLessThanCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32BranchLessThanExecutor =
    BranchLessThanExecutor<Rv32BranchAdapterExecutor, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32BranchLessThanChip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<Rv32BranchAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
