use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{BYTE_BITS, REGISTER_NUM_LIMBS};
use crate::adapters::MultAdapterAir;

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
pub(crate) mod trace;

pub type MultiplicationAir =
    VmAirWrapper<MultAdapterAir, MultiplicationCoreAir<REGISTER_NUM_LIMBS, BYTE_BITS>>;
pub type MultiplicationExecutor = MultiplicationCoreExecutor<REGISTER_NUM_LIMBS, BYTE_BITS>;
pub type MultiplicationChip<F> =
    VmChipWrapper<F, MultiplicationFiller<REGISTER_NUM_LIMBS, BYTE_BITS>>;
