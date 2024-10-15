use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32BranchAdapter};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualChip<F> = VmChipWrapper<F, Rv32BranchAdapter<F>, BranchEqualCoreChip<4>>;
