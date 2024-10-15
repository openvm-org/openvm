use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32BaseAluAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU256 module upon completion
pub type Rv32BaseAluChip<F> = VmChipWrapper<F, Rv32BaseAluAdapterChip<F>, BaseAluCoreChip<4, 8>>;
