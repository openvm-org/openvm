use crate::arch::{Rv32AluAdapter, VmChipWrapper};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU less than commands upon completion
pub type Rv32LessThanChip<F> = VmChipWrapper<F, Rv32AluAdapter<F>, LessThanCore<4, 8>>;
