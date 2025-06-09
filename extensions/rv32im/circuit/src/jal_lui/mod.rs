use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{Rv32CondRdWriteAdapterAir, Rv32CondRdWriteAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalLuiAir = VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir>;
pub type Rv32JalLuiStepWithAdapter = Rv32JalLuiStep<Rv32CondRdWriteAdapterStep>;
pub type Rv32JalLuiChip<F> =
    NewVmChipWrapper<F, Rv32JalLuiAir, Rv32JalLuiStepWithAdapter, MatrixRecordArena<F>>;
