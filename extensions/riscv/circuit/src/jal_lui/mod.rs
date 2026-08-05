use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::CondRdWriteAdapterAir;

mod core;
mod execution;
pub use core::*;

pub(crate) mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type JalLuiAir = VmAirWrapper<CondRdWriteAdapterAir, JalLuiCoreAir>;
pub type JalLuiChip<F> = VmChipWrapper<F, JalLuiFiller>;
