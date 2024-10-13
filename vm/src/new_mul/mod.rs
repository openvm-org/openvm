use crate::arch::{Rv32MultAdapter, VmChipWrapper};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Replace current uint_multiplication module upon completion
pub type Rv32MultiplicationChip<F> = VmChipWrapper<F, Rv32MultAdapter<F>, MultiplicationCore<4, 8>>;
