use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::Rv64RdWriteAdapterAir;

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64AuipcAir = VmAirWrapper<Rv64RdWriteAdapterAir, Rv64AuipcCoreAir>;
pub type Rv64AuipcChip<F> = VmChipWrapper<F, Rv64AuipcFiller>;
