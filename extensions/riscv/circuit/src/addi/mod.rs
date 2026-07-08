use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64ImmBaseAluU16AdapterAir, Rv64ImmBaseAluU16AdapterExecutor, Rv64ImmBaseAluU16AdapterFiller,
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

pub type Rv64AddIAir =
    VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, AddICoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64AddIExecutor =
    AddIExecutor<Rv64ImmBaseAluU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64AddIChip<F> =
    VmChipWrapper<F, AddIFiller<Rv64ImmBaseAluU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>>;
