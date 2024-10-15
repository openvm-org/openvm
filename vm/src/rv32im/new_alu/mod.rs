use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32AluAdapter};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU256 module upon completion
pub type Rv32ArithmeticLogicChip<F> =
    VmChipWrapper<F, Rv32AluAdapter<F>, ArithmeticLogicCoreChip<4, 8>>;
