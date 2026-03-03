use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::Rv32JalrAdapterAir;

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv32JalrAir = VmAirWrapper<Rv32JalrAdapterAir, Rv32JalrCoreAir>;
pub type Rv32JalrChip<F> = VmChipWrapper<F, Rv32JalrFiller>;
