use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, U16_CELL_SIZE};

use crate::{
    adapters::{LoadMultiByteAdapterAir, LoadMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH},
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

/// Cells overlapped by an odd-shift halfword load.
pub const LOAD_HALFWORD_OVERLAP_CELLS: usize = HALFWORD_ACCESS_WIDTH / U16_CELL_SIZE + 1;

pub type LoadHalfwordCoreAir = LoadCoreAir<HALFWORD_ACCESS_WIDTH, LOAD_HALFWORD_OVERLAP_CELLS>;
pub type LoadHalfwordFiller =
    LoadFiller<LoadMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH, LOAD_HALFWORD_OVERLAP_CELLS>;

pub type LoadHalfwordAir = VmAirWrapper<LoadMultiByteAdapterAir, LoadHalfwordCoreAir>;
pub type LoadHalfwordExecutor = LoadExecutor<HALFWORD_ACCESS_WIDTH>;
pub type LoadHalfwordChip<F> = VmChipWrapper<F, LoadHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
pub(crate) mod trace;
