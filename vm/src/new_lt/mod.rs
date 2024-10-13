use crate::arch::{VmChipWrapper, Rv32AluAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU less than commands upon completion
pub type Rv32LessThanChip<F> = VmChipWrapper<F, Rv32AluAdapter<F>, LessThanIntegration<4, 8>>;
