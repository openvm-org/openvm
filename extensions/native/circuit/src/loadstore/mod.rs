use axvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

use super::adapters::loadstore_native_adapter::{
    NativeLoadStoreAdapterAir, NativeLoadStoreAdapterChip,
};

pub type NativeLoadStoreAir<const NUM_CELLS: usize> =
    VmAirWrapper<NativeLoadStoreAdapterAir<NUM_CELLS>, NativeLoadStoreCoreAir<NUM_CELLS>>;
pub type NativeLoadStoreChip<F, const NUM_CELLS: usize> = VmChipWrapper<
    F,
    NativeLoadStoreAdapterChip<F, NUM_CELLS>,
    NativeLoadStoreCoreChip<F, NUM_CELLS>,
>;
