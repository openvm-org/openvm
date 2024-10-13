mod integration;

pub use integration::*;

use crate::arch::{VmChipWrapper, Rv32RdWriteAdapter};

#[cfg(test)]
mod tests;

pub type Rv32AuipcChip<F> = VmChipWrapper<F, Rv32RdWriteAdapter<F>, Rv32AuipcIntegration<F>>;
