use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::Rv32CondRdWriteAdapterAir;

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv32JalLuiAir = VmAirWrapper<Rv32CondRdWriteAdapterAir, Rv32JalLuiCoreAir>;
pub type Rv32JalLuiChip<F> = VmChipWrapper<F, Rv32JalLuiFiller>;
