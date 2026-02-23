use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::Rv64JalrAdapterAir;

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64JalrAir = VmAirWrapper<Rv64JalrAdapterAir, Rv64JalrCoreAir>;
pub type Rv64JalrChip<F> = VmChipWrapper<F, Rv64JalrFiller>;
