use crate::arch::{Rv32AluAdapter, VmChipWrapper};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current Shift256 module upon completion
pub type Rv32ShiftChip<F> = VmChipWrapper<F, Rv32AluAdapter<F>, ShiftCore<4, 8>>;
