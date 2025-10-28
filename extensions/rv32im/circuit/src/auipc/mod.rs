use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::Rv32RdWriteAdapterAir;

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv32AuipcAir = VmAirWrapper<Rv32RdWriteAdapterAir, Rv32AuipcCoreAir>;
pub type Rv32AuipcChip<F> = VmChipWrapper<F, Rv32AuipcFiller>;
