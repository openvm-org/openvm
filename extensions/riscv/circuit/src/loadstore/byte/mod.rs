use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor},
    loadstore::common::{LoadStoreExecutor, KIND_BYTE},
};

mod core;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64LoadStoreByteAir = VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreByteCoreAir>;
pub type Rv64LoadStoreByteExecutor = LoadStoreExecutor<Rv64LoadStoreAdapterExecutor, KIND_BYTE>;
pub type Rv64LoadStoreByteChip<F> = VmChipWrapper<F, LoadStoreByteFiller>;
