use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64RegBaseAluU16AdapterAir, Rv64RegBaseAluU16AdapterExecutor, Rv64RegBaseAluU16AdapterFiller,
    U16_BITS,
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
    VmAirWrapper<Rv64RegBaseAluU16AdapterAir, AddSubCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64AddSubExecutor =
    AddSubExecutor<Rv64RegBaseAluU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64AddSubChip<F> =
    VmChipWrapper<F, AddSubFiller<Rv64RegBaseAluU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>>;
