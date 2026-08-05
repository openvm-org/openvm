use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{StoreByteAdapterAir, BYTE_ACCESS_WIDTH},
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

pub type StoreByteAir = VmAirWrapper<StoreByteAdapterAir, StoreByteCoreAir>;
pub type StoreByteExecutor = StoreExecutor<BYTE_ACCESS_WIDTH>;
pub type StoreByteChip<F> = VmChipWrapper<F, StoreByteFiller>;
