use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64AddSubAdapterAir, Rv64AddSubAdapterExecutor, Rv64AddSubAdapterFiller, U16_BITS,
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
    VmAirWrapper<Rv64AddSubAdapterAir, AddSubCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64AddSubExecutor = AddSubExecutor<Rv64AddSubAdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64AddSubChip<F> =
    VmChipWrapper<F, AddSubFiller<Rv64AddSubAdapterFiller, BLOCK_FE_WIDTH, U16_BITS>>;
