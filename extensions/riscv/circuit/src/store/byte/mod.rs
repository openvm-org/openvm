use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64StoreByteAdapterAir, BYTE_ACCESS_WIDTH},
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
pub(crate) mod trace;

pub type Rv64StoreByteAir = VmAirWrapper<Rv64StoreByteAdapterAir, StoreByteCoreAir>;
pub type Rv64StoreByteExecutor = StoreExecutor<BYTE_ACCESS_WIDTH>;
pub type Rv64StoreByteChip<F> = VmChipWrapper<F, StoreByteFiller>;
