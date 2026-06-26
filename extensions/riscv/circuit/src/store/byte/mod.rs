use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64StoreAdapterAir, Rv64StoreAdapterExecutor},
    store::common::{StoreExecutor, KIND_BYTE},
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
pub type Rv64StoreByteExecutor = StoreExecutor<Rv64StoreAdapterExecutor, KIND_BYTE>;
pub type Rv64StoreByteChip<F> = VmChipWrapper<F, StoreByteFiller>;
