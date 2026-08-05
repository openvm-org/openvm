use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::RdWriteAdapterAir;

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

pub type AuipcAir = VmAirWrapper<RdWriteAdapterAir, AuipcCoreAir>;
pub type AuipcChip<F> = VmChipWrapper<F, AuipcFiller>;
