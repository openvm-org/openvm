use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{Rv64AddIAdapterAir, Rv64AddIAdapterExecutor, Rv64AddIAdapterFiller, U16_BITS};

mod core;
mod execution;
pub use core::*;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

pub type Rv64AddIAir = VmAirWrapper<Rv64AddIAdapterAir, AddICoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64AddIExecutor = AddIExecutor<Rv64AddIAdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64AddIChip<F> =
    VmChipWrapper<F, AddIFiller<Rv64AddIAdapterFiller, BLOCK_FE_WIDTH, U16_BITS>>;
