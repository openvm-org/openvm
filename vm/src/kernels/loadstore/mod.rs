use crate::arch::{VmAirWrapper, VmChipWrapper};

#[cfg(test)]
pub mod tests;

mod core;
pub use core::*;

use super::adapters::loadstore_native_adapter::{
    NativeLoadStoreAdapterAir, NativeLoadStoreAdapterChip,
};

pub type KernelLoadStoreAir<const NUM_CELLS: usize> =
    VmAirWrapper<NativeLoadStoreAdapterAir<NUM_CELLS>, KernelLoadStoreCoreAir<NUM_CELLS>>;
pub type KernelLoadStoreChip<F, const NUM_CELLS: usize> =
    VmChipWrapper<F, NativeLoadStoreAdapterChip<F, NUM_CELLS>, KernelLoadStoreCoreChip<NUM_CELLS>>;
