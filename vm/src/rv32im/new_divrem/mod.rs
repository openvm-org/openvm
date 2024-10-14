use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32MultAdapter};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Remove new_* prefix when completed
pub type Rv32DivRemChip<F> = VmChipWrapper<F, Rv32MultAdapter<F>, DivRemCoreChip<4, 8>>;
