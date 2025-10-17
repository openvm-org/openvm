use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{
    NativeLoadStoreAdapterAir, NativeLoadStoreAdapterExecutor, NativeLoadStoreAdapterFiller,
};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type NativeLoadStoreAir<const NUM_CELLS: usize> =
    VmAirWrapper<NativeLoadStoreAdapterAir<NUM_CELLS>, NativeLoadStoreCoreAir<NUM_CELLS>>;
pub type NativeLoadStoreExecutor<const NUM_CELLS: usize> =
    NativeLoadStoreCoreExecutor<NativeLoadStoreAdapterExecutor<NUM_CELLS>, NUM_CELLS>;
pub type NativeLoadStoreChip<F, const NUM_CELLS: usize> =
    VmChipWrapper<F, NativeLoadStoreCoreFiller<NativeLoadStoreAdapterFiller<NUM_CELLS>, NUM_CELLS>>;
