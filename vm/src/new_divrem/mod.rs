use crate::arch::{VmChipWrapper, Rv32MultAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Remove new_* prefix when completed
pub type Rv32DivRemChip<F> = VmChipWrapper<F, Rv32MultAdapter<F>, DivRemIntegration<4, 8>>;
