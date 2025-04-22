use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper, VmChipWrapper};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

use crate::adapters::loadstore_native_adapter::NativeLoadStoreAdapterStep;

use super::adapters::loadstore_native_adapter::{
    NativeLoadStoreAdapterAir, NativeLoadStoreAdapterChip,
};

pub type NativeLoadStoreAir<const NUM_CELLS: usize> =
    VmAirWrapper<NativeLoadStoreAdapterAir<NUM_CELLS>, NativeLoadStoreCoreAir<NUM_CELLS>>;
pub type NativeLoadStoreStepWithAdapter<const NUM_CELLS: usize> =
    NativeLoadStoreCoreStep<NativeLoadStoreAdapterStep<NUM_CELLS>>;
pub type NativeLoadStoreChip<F, const NUM_CELLS: usize> =
    NewVmChipWrapper<F, NativeLoadStoreAir<NUM_CELLS>, NativeLoadStoreStepWithAdapter<NUM_CELLS>>;
