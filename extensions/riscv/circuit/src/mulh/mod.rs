use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{BYTE_BITS, REGISTER_NUM_LIMBS};
use crate::adapters::MultAdapterAir;

mod core;
mod execution;
pub use core::*;

pub(crate) mod trace;
#[cfg(test)]
pub use trace::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type MulHAir = VmAirWrapper<MultAdapterAir, MulHCoreAir<REGISTER_NUM_LIMBS, BYTE_BITS>>;
pub type MulHExecutor = MulHCoreExecutor<REGISTER_NUM_LIMBS, BYTE_BITS>;
pub type MulHChip<F> = VmChipWrapper<F, MulHFiller<REGISTER_NUM_LIMBS, BYTE_BITS>>;
