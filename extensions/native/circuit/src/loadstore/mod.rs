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
pub type NativeLoadStoreStep<F, const NUM_CELLS: usize> =
    NativeLoadStoreCoreStep<NativeLoadStoreAdapterStep<NUM_CELLS>, F, NUM_CELLS>;
pub type NativeLoadStoreChip<F, const NUM_CELLS: usize> =
    NewVmChipWrapper<F, NativeLoadStoreAir<NUM_CELLS>, NativeLoadStoreStep<F, NUM_CELLS>>;
