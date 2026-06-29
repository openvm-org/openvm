use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller},
    store::{
        common::{StoreExecutor, KIND_WORD},
        width_aligned::{StoreWidthAlignedCoreAir, StoreWidthAlignedFiller},
    },
};

pub const WORD_STORE_CASES: usize = 2;
pub const WORD_STORE_SELECTOR_WIDTH: usize = 1;

pub type StoreWordCoreAir =
    StoreWidthAlignedCoreAir<KIND_WORD, WORD_STORE_CASES, WORD_STORE_SELECTOR_WIDTH>;
pub type StoreWordFiller = StoreWidthAlignedFiller<
    Rv64StoreAdapterFiller,
    KIND_WORD,
    WORD_STORE_CASES,
    WORD_STORE_SELECTOR_WIDTH,
>;

pub type Rv64StoreWordAir = VmAirWrapper<Rv64StoreAdapterAir, StoreWordCoreAir>;
pub type Rv64StoreWordExecutor = StoreExecutor<Rv64StoreAdapterExecutor, KIND_WORD>;
pub type Rv64StoreWordChip<F> = VmChipWrapper<F, StoreWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
