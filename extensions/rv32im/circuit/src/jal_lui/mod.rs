use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use crate::adapters::Rv32CondRdWriteAdapterAir;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalLuiAir = VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir>;
pub type Rv32JalLuiChip<F> = NewVmChipWrapper<F, Rv32JalLuiAir, Rv32JalLuiCoreChip>;
