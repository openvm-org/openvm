use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32MultAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current uint_multiplication module upon completion
pub type Rv32MultiplicationChip<F> =
    VmChipWrapper<F, Rv32MultAdapterChip<F>, MultiplicationCoreChip<4, 8>>;
