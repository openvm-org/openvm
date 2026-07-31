use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{Rv64BaseAluRegU16AdapterAir, U16_BITS};

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

pub type Rv64AddSubAir =
    VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<BLOCK_FE_WIDTH, U16_BITS, true>>;
pub type Rv64AddSubExecutor = AddSubExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64AddSubChip<F> = VmChipWrapper<F, AddSubFiller>;
