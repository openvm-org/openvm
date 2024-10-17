mod core;

pub use core::*;

use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32CondRdWriteAdapter};

#[cfg(test)]
mod tests;

pub type Rv32JalLuiChip<F> = VmChipWrapper<F, Rv32CondRdWriteAdapter<F>, Rv32JalLuiCoreChip>;
