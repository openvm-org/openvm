use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller},
    load::{
        common::{LoadExecutor, KIND_WORD},
        width_aligned::{LoadWidthAlignedCoreAir, LoadWidthAlignedFiller},
    },
};

pub const WORD_LOAD_CASES: usize = 2;
pub const WORD_LOAD_SELECTOR_WIDTH: usize = 1;

pub type LoadWordCoreAir =
    LoadWidthAlignedCoreAir<KIND_WORD, WORD_LOAD_CASES, WORD_LOAD_SELECTOR_WIDTH>;
pub type LoadWordFiller = LoadWidthAlignedFiller<
    Rv64LoadAdapterFiller,
    KIND_WORD,
    WORD_LOAD_CASES,
    WORD_LOAD_SELECTOR_WIDTH,
>;

pub type Rv64LoadWordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadWordCoreAir>;
pub type Rv64LoadWordExecutor = LoadExecutor<Rv64LoadAdapterExecutor, KIND_WORD>;
pub type Rv64LoadWordChip<F> = VmChipWrapper<F, LoadWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
