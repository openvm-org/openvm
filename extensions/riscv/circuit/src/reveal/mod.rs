use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};
use openvm_instructions::riscv::REGISTER_NUM_LIMBS;

mod adapter;
mod core;
mod execution;
pub(crate) mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub use core::*;

pub use adapter::*;
pub use execution::*;

pub(crate) const REVEAL_ACCESS_WIDTH: usize = REGISTER_NUM_LIMBS;
pub(crate) const REVEAL_VALUE_CELLS: usize = BLOCK_FE_WIDTH;

pub type RevealAir = VmAirWrapper<RevealAdapterAir, RevealCoreAir>;
pub type RevealChip<F> = VmChipWrapper<F, RevealFiller>;
