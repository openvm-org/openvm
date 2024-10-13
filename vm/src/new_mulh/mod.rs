use crate::arch::{VmChipWrapper, Rv32MultAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Remove new_* prefix when completed
pub type Rv32MulHChip<F> = VmChipWrapper<F, Rv32MultAdapter<F>, MulHIntegration<4, 8>>;
