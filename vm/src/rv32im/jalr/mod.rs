mod core;

pub use core::*;

use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32JalrAdapterChip};

#[cfg(test)]
mod tests;

pub type Rv32JalrChip<F> = VmChipWrapper<F, Rv32JalrAdapterChip<F>, Rv32JalrCoreChip>;
