use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, U16_CELL_SIZE};

use crate::{
    adapters::{
        Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, DOUBLEWORD_ACCESS_WIDTH,
    },
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

/// Cells overlapped by an odd-shift doubleword load.
pub const LOAD_DOUBLEWORD_OVERLAP_CELLS: usize = DOUBLEWORD_ACCESS_WIDTH / U16_CELL_SIZE + 1;

pub type LoadDoublewordCoreAir =
    LoadCoreAir<DOUBLEWORD_ACCESS_WIDTH, LOAD_DOUBLEWORD_OVERLAP_CELLS>;
pub type LoadDoublewordFiller = LoadFiller<
    Rv64LoadMultiByteAdapterFiller,
    DOUBLEWORD_ACCESS_WIDTH,
    LOAD_DOUBLEWORD_OVERLAP_CELLS,
>;

pub type Rv64LoadDoublewordAir = VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadDoublewordCoreAir>;
pub type Rv64LoadDoublewordExecutor = LoadExecutor<
    Rv64LoadMultiByteAdapterExecutor<DOUBLEWORD_ACCESS_WIDTH>,
    DOUBLEWORD_ACCESS_WIDTH,
>;
pub type Rv64LoadDoublewordChip<F> = VmChipWrapper<F, LoadDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
