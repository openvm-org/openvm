use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32BranchAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchLessThanChip<F> =
    VmChipWrapper<F, Rv32BranchAdapterChip<F>, BranchLessThanCoreChip<4, 8>>;
