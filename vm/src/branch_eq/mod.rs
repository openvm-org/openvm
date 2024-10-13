use crate::arch::{VmChipWrapper, Rv32BranchAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualChip<F> =
    VmChipWrapper<F, Rv32BranchAdapter<F>, BranchEqualIntegration<4>>;
