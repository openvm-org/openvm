use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32MultAdapter};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current uint_multiplication module upon completion
pub type Rv32MultiplicationChip<F> =
    VmChipWrapper<F, Rv32MultAdapter<F>, MultiplicationCoreChip<4, 8>>;
