use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, LOAD_WIDTH_WORD,
    },
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

pub const LOAD_WORD_SELECTOR_WIDTH: usize = 1;

pub type LoadWordCoreAir = LoadCoreAir<LOAD_WIDTH_WORD, LOAD_WORD_SELECTOR_WIDTH>;
pub type LoadWordFiller =
    LoadFiller<Rv64LoadAdapterFiller, LOAD_WIDTH_WORD, LOAD_WORD_SELECTOR_WIDTH>;

pub type Rv64LoadWordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadWordCoreAir>;
pub type Rv64LoadWordExecutor = LoadExecutor<Rv64LoadAdapterExecutor, LOAD_WIDTH_WORD>;
pub type Rv64LoadWordChip<F> = VmChipWrapper<F, LoadWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
