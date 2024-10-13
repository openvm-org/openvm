use crate::arch::{VmChipWrapper, Rv32BranchAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchLessThanChip<F> =
    VmChipWrapper<F, Rv32BranchAdapter<F>, BranchLessThanIntegration<4, 8>>;
