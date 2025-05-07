use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{Rv32JalrAdapterAir, Rv32JalrAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalrAir = VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir>;
pub type Rv32JalrStepWithAdapter = Rv32JalrStep<Rv32JalrAdapterStep>;
pub type Rv32JalrChip<F> = NewVmChipWrapper<F, Rv32JalrAir, Rv32JalrStepWithAdapter>;
