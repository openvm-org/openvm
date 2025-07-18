use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32BranchAdapterAir, Rv32BranchAdapterFiller, Rv32BranchAdapterStep};

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
pub type Rv32BranchLessThanChip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<Rv32BranchAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
