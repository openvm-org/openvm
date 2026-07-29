use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{Rv64BaseAluRegU16AdapterAir, U16_BITS};

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

pub type Rv64LessThanAir =
    VmAirWrapper<Rv64BaseAluRegU16AdapterAir, LessThanCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64LessThanExecutor = LessThanExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64LessThanChip<F> = VmChipWrapper<F, LessThanFiller>;
