mod integration;

pub use integration::*;

use crate::arch::{Rv32RdWriteAdapter, VmChipWrapper};

#[cfg(test)]
mod tests;

pub type Rv32AuipcChip<F> = VmChipWrapper<F, Rv32RdWriteAdapter<F>, Rv32AuipcCore<F>>;
