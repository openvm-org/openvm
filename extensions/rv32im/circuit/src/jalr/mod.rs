use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::Rv32JalrAdapterAir;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalrAir = VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir>;
pub type Rv32JalrChip<F> = VmChipWrapper<F, Rv32JalrFiller>;
