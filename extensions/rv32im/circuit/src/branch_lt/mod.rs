use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32BranchAdapterAir, Rv32BranchAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchLessThanAir = VmAirWrapper<
    Rv32BranchAdapterAir,
    BranchLessThanCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32BranchLessThanStep =
    BranchLessThanStep<Rv32BranchAdapterStep, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32BranchLessThanChip<F> =
    NewVmChipWrapper<F, Rv32BranchLessThanAir, Rv32BranchLessThanStep, MatrixRecordArena<F>>;
