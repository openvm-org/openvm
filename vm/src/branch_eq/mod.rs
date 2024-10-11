use crate::arch::{MachineChipWrapper, Rv32BranchAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualChip<F> =
    MachineChipWrapper<F, Rv32BranchAdapter<F>, BranchEqualIntegration<4>>;
