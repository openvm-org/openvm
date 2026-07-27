use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS};
use crate::adapters::Rv64MultAdapterAir;

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

pub type Rv64MultiplicationAir = VmAirWrapper<
    Rv64MultAdapterAir,
    MultiplicationCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
>;
pub type Rv64MultiplicationExecutor =
    MultiplicationExecutor<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>;
pub type Rv64MultiplicationChip<F> =
    VmChipWrapper<F, MultiplicationFiller<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>>;
