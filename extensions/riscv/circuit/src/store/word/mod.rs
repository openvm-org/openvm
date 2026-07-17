use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, U16_CELL_SIZE};

use crate::{
    adapters::{
        Rv64StoreMultiByteAdapterAir, Rv64StoreMultiByteAdapterExecutor,
        Rv64StoreMultiByteAdapterFiller, WORD_ACCESS_WIDTH,
    },
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

/// Source register cells decomposed on an odd-shift word store.
pub const STORE_WORD_VALUE_CELLS: usize = WORD_ACCESS_WIDTH / U16_CELL_SIZE;

pub type StoreWordCoreAir = StoreCoreAir<WORD_ACCESS_WIDTH, STORE_WORD_VALUE_CELLS>;
pub type StoreWordFiller =
    StoreFiller<Rv64StoreMultiByteAdapterFiller, WORD_ACCESS_WIDTH, STORE_WORD_VALUE_CELLS>;

pub type Rv64StoreWordAir = VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreWordCoreAir>;
pub type Rv64StoreWordExecutor =
    StoreExecutor<Rv64StoreMultiByteAdapterExecutor<WORD_ACCESS_WIDTH>, WORD_ACCESS_WIDTH>;
pub type Rv64StoreWordChip<F> = VmChipWrapper<F, StoreWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
