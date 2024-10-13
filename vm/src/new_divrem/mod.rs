use crate::arch::{Rv32MultAdapter, VmChipWrapper};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Remove new_* prefix when completed
pub type Rv32DivRemChip<F> = VmChipWrapper<F, Rv32MultAdapter<F>, DivRemCore<4, 8>>;
