use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, U16_CELL_SIZE};

use crate::{
    adapters::{
        Rv64StoreMultiByteAdapterAir, Rv64StoreMultiByteAdapterExecutor,
        Rv64StoreMultiByteAdapterFiller, DOUBLEWORD_ACCESS_WIDTH,
    },
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

/// Source register cells decomposed on an odd-shift doubleword store.
pub const STORE_DOUBLEWORD_VALUE_CELLS: usize = DOUBLEWORD_ACCESS_WIDTH / U16_CELL_SIZE;

pub type StoreDoublewordCoreAir =
    StoreCoreAir<DOUBLEWORD_ACCESS_WIDTH, STORE_DOUBLEWORD_VALUE_CELLS>;
pub type StoreDoublewordFiller = StoreFiller<
    Rv64StoreMultiByteAdapterFiller,
    DOUBLEWORD_ACCESS_WIDTH,
    STORE_DOUBLEWORD_VALUE_CELLS,
>;

pub type Rv64StoreDoublewordAir =
    VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreDoublewordCoreAir>;
pub type Rv64StoreDoublewordExecutor = StoreExecutor<
    Rv64StoreMultiByteAdapterExecutor<DOUBLEWORD_ACCESS_WIDTH>,
    DOUBLEWORD_ACCESS_WIDTH,
>;
pub type Rv64StoreDoublewordChip<F> = VmChipWrapper<F, StoreDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
