mod integration;

pub use integration::*;

use crate::arch::{MachineChipWrapper, Rv32JalrAdapter};

#[cfg(test)]
mod tests;

pub type Rv32JalrChip<F> = MachineChipWrapper<F, Rv32JalrAdapter<F>, Rv32JalrIntegration<F>>;
