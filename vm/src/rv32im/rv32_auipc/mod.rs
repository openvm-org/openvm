mod core;

pub use core::*;

use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32RdWriteAdapter};

#[cfg(test)]
mod tests;

pub type Rv32AuipcChip<F> = VmChipWrapper<F, Rv32RdWriteAdapter<F>, Rv32AuipcCoreChip>;
