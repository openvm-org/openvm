use crate::arch::{Rv32AluAdapter, VmChipWrapper};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU256 module upon completion
pub type Rv32ArithmeticLogicChip<F> =
    VmChipWrapper<F, Rv32AluAdapter<F>, ArithmeticLogicCore<4, 8>>;
