mod core;

pub use core::*;

use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::RV64_REGISTER_NUM_LIMBS;
use crate::adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "aot")]
mod aot;

#[cfg(all(test, any()))] // TODO: port tests to RV64
mod tests;

pub type Rv64LoadStoreAir =
    VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreCoreAir<RV64_REGISTER_NUM_LIMBS>>;
pub type Rv64LoadStoreExecutor =
    LoadStoreExecutor<Rv64LoadStoreAdapterExecutor, RV64_REGISTER_NUM_LIMBS>;
pub type Rv64LoadStoreChip<F> = VmChipWrapper<F, LoadStoreFiller>;
