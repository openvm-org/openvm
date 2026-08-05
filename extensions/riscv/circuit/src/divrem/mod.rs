use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{BYTE_BITS, REGISTER_NUM_LIMBS};
use crate::adapters::{MultAdapterAir, MultAdapterFiller};

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

pub type DivRemAir = VmAirWrapper<MultAdapterAir, DivRemCoreAir<REGISTER_NUM_LIMBS, BYTE_BITS>>;
pub type DivRemExecutor = DivRemCoreExecutor<REGISTER_NUM_LIMBS, BYTE_BITS>;
pub type DivRemChip<F> =
    VmChipWrapper<F, DivRemFiller<MultAdapterFiller, REGISTER_NUM_LIMBS, BYTE_BITS>>;
