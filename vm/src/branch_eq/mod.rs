use crate::arch::{Rv32BranchAdapter, VmChipWrapper};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualChip<F> = VmChipWrapper<F, Rv32BranchAdapter<F>, BranchEqualCoreChip<4>>;
