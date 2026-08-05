use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::JalrAdapterAir;

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

pub type JalrAir = VmAirWrapper<JalrAdapterAir, JalrCoreAir>;
pub type JalrChip<F> = VmChipWrapper<F, JalrFiller>;
