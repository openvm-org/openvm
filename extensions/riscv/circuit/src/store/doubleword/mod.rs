use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, U16_CELL_SIZE};

use crate::{
    adapters::{StoreMultiByteAdapterAir, StoreMultiByteAdapterFiller, DOUBLEWORD_ACCESS_WIDTH},
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

/// Source register cells decomposed on an odd-shift doubleword store.
pub const STORE_DOUBLEWORD_VALUE_CELLS: usize = DOUBLEWORD_ACCESS_WIDTH / U16_CELL_SIZE;

pub type StoreDoublewordCoreAir =
    StoreCoreAir<DOUBLEWORD_ACCESS_WIDTH, STORE_DOUBLEWORD_VALUE_CELLS>;
pub type StoreDoublewordFiller =
    StoreFiller<StoreMultiByteAdapterFiller, DOUBLEWORD_ACCESS_WIDTH, STORE_DOUBLEWORD_VALUE_CELLS>;

pub type StoreDoublewordAir = VmAirWrapper<StoreMultiByteAdapterAir, StoreDoublewordCoreAir>;
pub type StoreDoublewordExecutor = StoreExecutor<DOUBLEWORD_ACCESS_WIDTH>;
pub type StoreDoublewordChip<F> = VmChipWrapper<F, StoreDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
pub(crate) mod trace;
