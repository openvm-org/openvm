use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::Rv64CondRdWriteAdapterAir;

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64JalLuiAir = VmAirWrapper<Rv64CondRdWriteAdapterAir, Rv64JalLuiCoreAir>;
pub type Rv64JalLuiChip<F> = VmChipWrapper<F, Rv64JalLuiFiller>;
