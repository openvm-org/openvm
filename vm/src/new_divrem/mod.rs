use crate::arch::{Rv32MultAdapter, VmChipWrapper};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Remove new_* prefix when completed
pub type Rv32DivRemChip<F> = VmChipWrapper<F, Rv32MultAdapter<F>, DivRemCore<4, 8>>;
