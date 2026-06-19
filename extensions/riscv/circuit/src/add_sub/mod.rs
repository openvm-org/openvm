use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64BaseAluU16AdapterAir, Rv64BaseAluU16AdapterExecutor, Rv64BaseAluU16AdapterFiller, U16_BITS,
};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64AddSubAir =
    VmAirWrapper<Rv64BaseAluU16AdapterAir, AddSubCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64AddSubExecutor =
    AddSubExecutor<Rv64BaseAluU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64AddSubChip<F> =
    VmChipWrapper<F, AddSubFiller<Rv64BaseAluU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>>;
