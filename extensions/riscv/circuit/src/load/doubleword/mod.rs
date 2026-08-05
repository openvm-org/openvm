use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, U16_CELL_SIZE};

use crate::{
    adapters::{LoadMultiByteAdapterAir, LoadMultiByteAdapterFiller, DOUBLEWORD_ACCESS_WIDTH},
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

/// Cells overlapped by an odd-shift doubleword load.
pub const LOAD_DOUBLEWORD_OVERLAP_CELLS: usize = DOUBLEWORD_ACCESS_WIDTH / U16_CELL_SIZE + 1;

pub type LoadDoublewordCoreAir =
    LoadCoreAir<DOUBLEWORD_ACCESS_WIDTH, LOAD_DOUBLEWORD_OVERLAP_CELLS>;
pub type LoadDoublewordFiller =
    LoadFiller<LoadMultiByteAdapterFiller, DOUBLEWORD_ACCESS_WIDTH, LOAD_DOUBLEWORD_OVERLAP_CELLS>;

pub type LoadDoublewordAir = VmAirWrapper<LoadMultiByteAdapterAir, LoadDoublewordCoreAir>;
pub type LoadDoublewordExecutor = LoadExecutor<DOUBLEWORD_ACCESS_WIDTH>;
pub type LoadDoublewordChip<F> = VmChipWrapper<F, LoadDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
pub(crate) mod trace;
