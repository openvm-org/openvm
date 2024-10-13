mod core;

pub use core::*;

use crate::arch::{Rv32JalrAdapter, VmChipWrapper};

#[cfg(test)]
mod tests;

pub type Rv32JalrChip<F> = VmChipWrapper<F, Rv32JalrAdapter<F>, Rv32JalrCore<F>>;
