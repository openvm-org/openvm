use crate::arch::{Rv32MultAdapter, VmChipWrapper};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

// TODO: Replace current uint_multiplication module upon completion
pub type Rv32MultiplicationChip<F> =
    VmChipWrapper<F, Rv32MultAdapter<F>, MultiplicationCoreChip<4, 8>>;
