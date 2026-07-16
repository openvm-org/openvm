use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64StoreMultiByteAdapterAir, Rv64StoreMultiByteAdapterExecutor,
        Rv64StoreMultiByteAdapterFiller, STORE_WIDTH_WORD,
    },
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

/// Source register cells decomposed on an odd-shift word store: `STORE_WIDTH_WORD / 2`.
pub const STORE_WORD_VALUE_CELLS: usize = STORE_WIDTH_WORD / 2;

pub type StoreWordCoreAir = StoreCoreAir<STORE_WIDTH_WORD, STORE_WORD_VALUE_CELLS>;
pub type StoreWordFiller =
    StoreFiller<Rv64StoreMultiByteAdapterFiller, STORE_WIDTH_WORD, STORE_WORD_VALUE_CELLS>;

pub type Rv64StoreWordAir = VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreWordCoreAir>;
pub type Rv64StoreWordExecutor =
    StoreExecutor<Rv64StoreMultiByteAdapterExecutor<STORE_WIDTH_WORD>, STORE_WIDTH_WORD>;
pub type Rv64StoreWordChip<F> = VmChipWrapper<F, StoreWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
