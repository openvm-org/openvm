use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, LOAD_WIDTH_WORD,
    },
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

/// Cells overlapped by an odd-shift word load: `LOAD_WIDTH_WORD / 2 + 1`.
pub const LOAD_WORD_OVERLAP_CELLS: usize = LOAD_WIDTH_WORD / 2 + 1;

pub type LoadWordCoreAir = LoadCoreAir<LOAD_WIDTH_WORD, LOAD_WORD_OVERLAP_CELLS>;
pub type LoadWordFiller =
    LoadFiller<Rv64LoadMultiByteAdapterFiller, LOAD_WIDTH_WORD, LOAD_WORD_OVERLAP_CELLS>;

pub type Rv64LoadWordAir = VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadWordCoreAir>;
pub type Rv64LoadWordExecutor =
    LoadExecutor<Rv64LoadMultiByteAdapterExecutor<LOAD_WIDTH_WORD>, LOAD_WIDTH_WORD>;
pub type Rv64LoadWordChip<F> = VmChipWrapper<F, LoadWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
