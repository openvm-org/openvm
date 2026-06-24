use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor},
    loadstore::{
        aligned::{LoadStoreAlignedCoreAir, LoadStoreAlignedFiller},
        common::{LoadStoreExecutor, KIND_WORD},
    },
};

pub const WORD_CASES: usize = 4;
pub const WORD_SELECTOR_WIDTH: usize = 2;

pub type LoadStoreWordCoreAir = LoadStoreAlignedCoreAir<KIND_WORD, WORD_CASES, WORD_SELECTOR_WIDTH>;
pub type LoadStoreWordFiller = LoadStoreAlignedFiller<
    crate::adapters::Rv64LoadStoreAdapterFiller,
    KIND_WORD,
    WORD_CASES,
    WORD_SELECTOR_WIDTH,
>;

pub type Rv64LoadStoreWordAir = VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreWordCoreAir>;
pub type Rv64LoadStoreWordExecutor = LoadStoreExecutor<Rv64LoadStoreAdapterExecutor, KIND_WORD>;
pub type Rv64LoadStoreWordChip<F> = VmChipWrapper<F, LoadStoreWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
