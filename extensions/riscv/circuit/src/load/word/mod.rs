use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, WORD_ACCESS_WIDTH,
    },
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

/// Cells overlapped by an odd-shift word load: `WORD_ACCESS_WIDTH / 2 + 1`.
pub const LOAD_WORD_OVERLAP_CELLS: usize = WORD_ACCESS_WIDTH / 2 + 1;

pub type LoadWordCoreAir = LoadCoreAir<WORD_ACCESS_WIDTH, LOAD_WORD_OVERLAP_CELLS>;
pub type LoadWordFiller =
    LoadFiller<Rv64LoadMultiByteAdapterFiller, WORD_ACCESS_WIDTH, LOAD_WORD_OVERLAP_CELLS>;

pub type Rv64LoadWordAir = VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadWordCoreAir>;
pub type Rv64LoadWordExecutor =
    LoadExecutor<Rv64LoadMultiByteAdapterExecutor<WORD_ACCESS_WIDTH>, WORD_ACCESS_WIDTH>;
pub type Rv64LoadWordChip<F> = VmChipWrapper<F, LoadWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
