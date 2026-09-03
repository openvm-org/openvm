use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, U16_CELL_SIZE};

use crate::{
    adapters::{StoreMultiByteAdapterAir, StoreMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH},
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

/// Source register cells decomposed on an odd-shift halfword store.
pub const STORE_HALFWORD_VALUE_CELLS: usize = HALFWORD_ACCESS_WIDTH / U16_CELL_SIZE;

pub type StoreHalfwordCoreAir = StoreCoreAir<HALFWORD_ACCESS_WIDTH, STORE_HALFWORD_VALUE_CELLS>;
pub type StoreHalfwordFiller =
    StoreFiller<StoreMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH, STORE_HALFWORD_VALUE_CELLS>;

pub type StoreHalfwordAir = VmAirWrapper<StoreMultiByteAdapterAir, StoreHalfwordCoreAir>;
pub type StoreHalfwordExecutor = StoreExecutor<HALFWORD_ACCESS_WIDTH>;
pub type StoreHalfwordChip<F> = VmChipWrapper<F, StoreHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
pub(crate) mod trace;
