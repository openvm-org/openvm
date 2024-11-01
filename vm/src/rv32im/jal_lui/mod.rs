mod core;

pub use core::*;

use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32CondRdWriteAdapterChip};

#[cfg(test)]
mod tests;

pub type Rv32JalLuiChip<F> = VmChipWrapper<F, Rv32CondRdWriteAdapterChip<F>, Rv32JalLuiCoreChip>;
