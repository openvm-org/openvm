use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{BaseAluRegU16AdapterAir, U16_BITS};

mod core;
mod execution;
pub(crate) mod trace;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type AddSubAir =
    VmAirWrapper<BaseAluRegU16AdapterAir, AddSubCoreAir<BLOCK_FE_WIDTH, U16_BITS, true>>;
pub type AddSubExecutor = AddSubCoreExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type AddSubChip<F> = VmChipWrapper<F, AddSubFiller>;
