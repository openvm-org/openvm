use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

use super::adapters::loadstore_native_adapter::{
    NativeLoadStoreAdapterAir, NativeLoadStoreAdapterStep,
};

pub type NativeLoadStoreAir<const NUM_CELLS: usize> =
    VmAirWrapper<NativeLoadStoreAdapterAir<NUM_CELLS>, NativeLoadStoreCoreAir<NUM_CELLS>>;
pub type NativeLoadStoreStep<const NUM_CELLS: usize> =
    NativeLoadStoreCoreStep<NativeLoadStoreAdapterStep<NUM_CELLS>, NUM_CELLS>;
pub type NativeLoadStoreChip<F, const NUM_CELLS: usize> =
    NewVmChipWrapper<F, NativeLoadStoreAir<NUM_CELLS>, NativeLoadStoreStep<NUM_CELLS>>;
