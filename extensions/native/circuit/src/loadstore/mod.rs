use axvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

use super::adapters::loadstore_native_adapter::{
    NativeLoadStoreAdapterAir, NativeLoadStoreAdapterChip,
};

pub type KernelLoadStoreAir<const NUM_CELLS: usize> =
    VmAirWrapper<NativeLoadStoreAdapterAir<NUM_CELLS>, KernelLoadStoreCoreAir<NUM_CELLS>>;
pub type KernelLoadStoreChip<F, const NUM_CELLS: usize> = VmChipWrapper<
    F,
    NativeLoadStoreAdapterChip<F, NUM_CELLS>,
    KernelLoadStoreCoreChip<F, NUM_CELLS>,
>;
