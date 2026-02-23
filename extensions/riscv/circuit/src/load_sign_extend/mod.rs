use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "aot")]
mod aot;

#[cfg(test)]
mod tests;

pub type Rv64LoadSignExtendAir = VmAirWrapper<
    Rv64LoadStoreAdapterAir,
    LoadSignExtendCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv64LoadSignExtendExecutor =
    LoadSignExtendExecutor<Rv64LoadStoreAdapterExecutor, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv64LoadSignExtendChip<F> = VmChipWrapper<F, LoadSignExtendFiller>;
