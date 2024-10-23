use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32MultAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Remove new_* prefix when completed
pub type Rv32DivRemChip<F> = VmChipWrapper<F, Rv32MultAdapterChip<F>, DivRemCoreChip<4, 8>>;
