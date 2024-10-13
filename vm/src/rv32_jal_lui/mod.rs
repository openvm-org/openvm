mod core;

pub use core::*;

use crate::arch::{Rv32RdWriteAdapter, VmChipWrapper};

#[cfg(test)]
mod tests;

pub type Rv32JalLuiChip<F> = VmChipWrapper<F, Rv32RdWriteAdapter<F>, Rv32JalLuiCoreChip<F>>;
