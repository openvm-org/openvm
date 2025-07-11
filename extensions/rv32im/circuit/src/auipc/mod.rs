use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{Rv32RdWriteAdapterAir, Rv32RdWriteAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32AuipcAir = VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir>;
pub type Rv32AuipcStepWithAdapter = Rv32AuipcStep<Rv32RdWriteAdapterStep>;
pub type Rv32AuipcChip<F> =
    NewVmChipWrapper<F, Rv32AuipcAir, Rv32AuipcStepWithAdapter, MatrixRecordArena<F>>;
