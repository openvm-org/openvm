use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use crate::adapters::Rv32RdWriteAdapterAir;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32AuipcAir = VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir>;
pub type Rv32AuipcChip<F> = NewVmChipWrapper<F, Rv32AuipcAir, Rv32AuipcCoreChip>;
