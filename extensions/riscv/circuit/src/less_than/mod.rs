use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{BaseAluRegU16AdapterAir, U16_BITS};

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

pub type LessThanAir =
    VmAirWrapper<BaseAluRegU16AdapterAir, LessThanCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type LessThanExecutor = LessThanCoreExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type LessThanChip<F> = VmChipWrapper<F, LessThanFiller>;
