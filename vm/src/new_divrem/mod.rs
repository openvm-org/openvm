use crate::arch::{MachineChipWrapper, Rv32MultAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Remove new_* prefix when completed
pub type Rv32DivRemChip<F> = MachineChipWrapper<F, Rv32MultAdapter<F>, DivRemIntegration<4, 8>>;
