use crate::arch::{Rv32BranchAdapter, VmChipWrapper};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualChip<F> = VmChipWrapper<F, Rv32BranchAdapter<F>, BranchEqualCore<4>>;
