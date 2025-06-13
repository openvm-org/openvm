use crate::adapters::{NativeLoadStoreAdapterAir, NativeLoadStoreAdapterStep};
use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type NativeLoadStoreAir<const NUM_CELLS: usize> =
    VmAirWrapper<NativeLoadStoreAdapterAir<NUM_CELLS>, NativeLoadStoreCoreAir<NUM_CELLS>>;
pub type NativeLoadStoreStep<F, const NUM_CELLS: usize> =
    NativeLoadStoreCoreStep<NativeLoadStoreAdapterStep<NUM_CELLS>, F, NUM_CELLS>;
pub type NativeLoadStoreChip<F, const NUM_CELLS: usize> = NewVmChipWrapper<
    F,
    NativeLoadStoreAir<NUM_CELLS>,
    NativeLoadStoreStep<F, NUM_CELLS>,
    MatrixRecordArena<F>,
>;
