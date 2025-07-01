use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{Rv32JalrAdapterAir, Rv32JalrAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalrAir = VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir>;
