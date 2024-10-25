use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32BranchAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualChip<F> =
    VmChipWrapper<F, Rv32BranchAdapterChip<F>, BranchEqualCoreChip<RV32_REGISTER_NUM_LIMBS>>;
