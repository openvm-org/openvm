mod integration;

pub use integration::*;

use crate::arch::{VmChipWrapper, Rv32JalrAdapter};

#[cfg(test)]
mod tests;

pub type Rv32JalrChip<F> = VmChipWrapper<F, Rv32JalrAdapter<F>, Rv32JalrIntegration<F>>;
