use crate::arch::{MachineChipWrapper, Rv32AluAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU256 module upon completion
pub type Rv32ArithmeticLogicChip<F> =
    MachineChipWrapper<F, Rv32AluAdapter<F>, ArithmeticLogicIntegration<4, 8>>;
