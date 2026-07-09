use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, STORE_WIDTH_BYTE},
    store::common::StoreExecutor,
};

mod core;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64StoreByteAir = VmAirWrapper<Rv64StoreAdapterAir, StoreByteCoreAir>;
pub type Rv64StoreByteExecutor = StoreExecutor<Rv64StoreAdapterExecutor, STORE_WIDTH_BYTE>;
pub type Rv64StoreByteChip<F> = VmChipWrapper<F, StoreByteFiller>;
