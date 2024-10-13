use crate::arch::{VmChipWrapper, Rv32AluAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU256 module upon completion
pub type Rv32ArithmeticLogicChip<F> =
    VmChipWrapper<F, Rv32AluAdapter<F>, ArithmeticLogicIntegration<4, 8>>;
