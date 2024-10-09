use crate::arch::{MachineChipWrapper, Rv32AluAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU less than commands upon completion
pub type Rv32LessThanChip<F> = MachineChipWrapper<F, Rv32AluAdapter<F>, LessThanIntegration<4, 8>>;
