use crate::arch::{MachineChipWrapper, Rv32AluAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Replace current Shift256 module upon completion
pub type Rv32ShiftChip<F> = MachineChipWrapper<F, Rv32AluAdapter<F>, ShiftIntegration<4, 8>>;
