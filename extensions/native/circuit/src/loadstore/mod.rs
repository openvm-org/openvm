use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{NativeLoadStoreAdapterAir, NativeLoadStoreAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type NativeLoadStoreAir<const NUM_CELLS: usize> =
    VmAirWrapper<NativeLoadStoreAdapterAir<NUM_CELLS>, NativeLoadStoreCoreAir<NUM_CELLS>>;
pub type NativeLoadStoreStep<const NUM_CELLS: usize> =
    NativeLoadStoreCoreStep<NativeLoadStoreAdapterStep<NUM_CELLS>, NUM_CELLS>;
pub type NativeLoadStoreChip<F, const NUM_CELLS: usize> = NewVmChipWrapper<
    F,
    NativeLoadStoreAir<NUM_CELLS>,
    NativeLoadStoreStep<NUM_CELLS>,
    MatrixRecordArena<F>,
>;
