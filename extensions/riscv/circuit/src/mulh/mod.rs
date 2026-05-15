use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv64MultAdapterAir, Rv64MultAdapterExecutor, Rv64MultAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64MulHAir =
    VmAirWrapper<Rv64MultAdapterAir, MulHCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>>;
pub type Rv64MulHExecutor =
    MulHExecutor<Rv64MultAdapterExecutor, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>;
pub type Rv64MulHChip<F> =
    VmChipWrapper<F, MulHFiller<Rv64MultAdapterFiller, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>>;
