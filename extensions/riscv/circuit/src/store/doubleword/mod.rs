use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

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

/// Source register cells decomposed on an odd-shift doubleword store: `DOUBLEWORD_ACCESS_WIDTH /
/// 2`.
pub const STORE_DOUBLEWORD_VALUE_CELLS: usize = DOUBLEWORD_ACCESS_WIDTH / 2;

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
