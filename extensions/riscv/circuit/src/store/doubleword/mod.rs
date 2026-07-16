use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64StoreMultiByteAdapterAir, Rv64StoreMultiByteAdapterExecutor,
        Rv64StoreMultiByteAdapterFiller, STORE_WIDTH_DOUBLEWORD,
    },
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

pub const STORE_DOUBLEWORD_SELECTOR_WIDTH: usize = 3;
/// Source register cells decomposed on an odd-shift doubleword store: `STORE_WIDTH_DOUBLEWORD / 2`.
pub const STORE_DOUBLEWORD_VALUE_CELLS: usize = 4;

pub type StoreDoublewordCoreAir = StoreCoreAir<
    STORE_WIDTH_DOUBLEWORD,
    STORE_DOUBLEWORD_SELECTOR_WIDTH,
    STORE_DOUBLEWORD_VALUE_CELLS,
>;
pub type StoreDoublewordFiller = StoreFiller<
    Rv64StoreMultiByteAdapterFiller,
    STORE_WIDTH_DOUBLEWORD,
    STORE_DOUBLEWORD_SELECTOR_WIDTH,
    STORE_DOUBLEWORD_VALUE_CELLS,
>;

pub type Rv64StoreDoublewordAir =
    VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreDoublewordCoreAir>;
pub type Rv64StoreDoublewordExecutor = StoreExecutor<
    Rv64StoreMultiByteAdapterExecutor<STORE_WIDTH_DOUBLEWORD>,
    STORE_WIDTH_DOUBLEWORD,
>;
pub type Rv64StoreDoublewordChip<F> = VmChipWrapper<F, StoreDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
