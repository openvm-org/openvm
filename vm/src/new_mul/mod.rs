use crate::arch::{MachineChipWrapper, Rv32MultAdapter};

mod integration;
pub use integration::*;

#[cfg(test)]
mod tests;

// TODO: Replace current uint_multiplication module upon completion
pub type Rv32MultiplicationChip<F> =
    MachineChipWrapper<F, Rv32MultAdapter<F>, MultiplicationIntegration<4, 8>>;
