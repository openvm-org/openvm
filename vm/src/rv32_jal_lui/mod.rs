mod integration;

pub use integration::*;

use crate::arch::{VmChipWrapper, Rv32RdWriteAdapter};

#[cfg(test)]
mod tests;

pub type Rv32JalLuiChip<F> = VmChipWrapper<F, Rv32RdWriteAdapter<F>, Rv32JalLuiIntegration<F>>;
