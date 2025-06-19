use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32BranchAdapterAir, Rv32BranchAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchLessThanChip = BranchLessThanStep<
    Rv32BranchAdapterAir,
    Rv32BranchAdapterStep,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
