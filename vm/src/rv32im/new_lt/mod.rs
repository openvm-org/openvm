use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32AluAdapter};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current ALU less than commands upon completion
pub type Rv32LessThanChip<F> = VmChipWrapper<F, Rv32AluAdapter<F>, LessThanCoreChip<4, 8>>;
