use crate::arch::{Rv32BranchAdapter, VmChipWrapper};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchLessThanChip<F> =
    VmChipWrapper<F, Rv32BranchAdapter<F>, BranchLessThanCore<4, 8>>;
