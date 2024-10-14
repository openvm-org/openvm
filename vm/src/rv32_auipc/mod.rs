mod core;

pub use core::*;

use crate::arch::{Rv32RdWriteAdapter, VmChipWrapper};

#[cfg(test)]
mod tests;

pub type Rv32AuipcChip<F> = VmChipWrapper<F, Rv32RdWriteAdapter<F>, Rv32AuipcCoreChip<F>>;
