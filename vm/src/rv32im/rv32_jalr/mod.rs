mod core;

pub use core::*;

use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32JalrAdapter};

#[cfg(test)]
mod tests;

pub type Rv32JalrChip<F> = VmChipWrapper<F, Rv32JalrAdapter<F>, Rv32JalrCoreChip>;
