mod core;

pub use core::*;

use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterExecutor};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreAir =
    VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv32LoadStoreExecutor =
    LoadStoreExecutor<Rv32LoadStoreAdapterExecutor, RV32_REGISTER_NUM_LIMBS>;
pub type Rv32LoadStoreChip<F> = VmChipWrapper<F, LoadStoreFiller>;
