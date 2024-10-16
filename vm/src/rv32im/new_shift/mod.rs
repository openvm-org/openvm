use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32BaseAluAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current Shift256 module upon completion
pub type Rv32ShiftChip<F> = VmChipWrapper<F, Rv32BaseAluAdapterChip<F>, ShiftCoreChip<4, 8>>;
