use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, U16_CELL_SIZE};

use crate::{
    adapters::{LoadMultiByteAdapterAir, LoadMultiByteAdapterFiller, WORD_ACCESS_WIDTH},
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

/// Cells overlapped by an odd-shift word load.
pub const LOAD_WORD_OVERLAP_CELLS: usize = WORD_ACCESS_WIDTH / U16_CELL_SIZE + 1;

pub type LoadWordCoreAir = LoadCoreAir<WORD_ACCESS_WIDTH, LOAD_WORD_OVERLAP_CELLS>;
pub type LoadWordFiller =
    LoadFiller<LoadMultiByteAdapterFiller, WORD_ACCESS_WIDTH, LOAD_WORD_OVERLAP_CELLS>;

pub type LoadWordAir = VmAirWrapper<LoadMultiByteAdapterAir, LoadWordCoreAir>;
pub type LoadWordExecutor = LoadExecutor<WORD_ACCESS_WIDTH>;
pub type LoadWordChip<F> = VmChipWrapper<F, LoadWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
pub(crate) mod trace;
