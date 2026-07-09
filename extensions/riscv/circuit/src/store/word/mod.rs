use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller, STORE_WIDTH_WORD,
    },
    store::{
        common::StoreExecutor,
        core::{StoreWidthAlignedCoreAir, StoreWidthAlignedFiller},
    },
};

pub const STORE_WORD_NUM_CASES: usize = 2;
pub const STORE_WORD_SELECTOR_WIDTH: usize = 1;

pub type StoreWordCoreAir =
    StoreWidthAlignedCoreAir<STORE_WIDTH_WORD, STORE_WORD_NUM_CASES, STORE_WORD_SELECTOR_WIDTH>;
pub type StoreWordFiller = StoreWidthAlignedFiller<
    Rv64StoreAdapterFiller,
    STORE_WIDTH_WORD,
    STORE_WORD_NUM_CASES,
    STORE_WORD_SELECTOR_WIDTH,
>;

pub type Rv64StoreWordAir = VmAirWrapper<Rv64StoreAdapterAir, StoreWordCoreAir>;
pub type Rv64StoreWordExecutor = StoreExecutor<Rv64StoreAdapterExecutor, STORE_WIDTH_WORD>;
pub type Rv64StoreWordChip<F> = VmChipWrapper<F, StoreWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
