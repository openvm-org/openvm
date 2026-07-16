use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, LOAD_WIDTH_DOUBLEWORD,
    },
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

pub const LOAD_DOUBLEWORD_SELECTOR_WIDTH: usize = 3;
/// Cells overlapped by an odd-shift doubleword load: `LOAD_WIDTH_DOUBLEWORD / 2 + 1`.
pub const LOAD_DOUBLEWORD_OVERLAP_CELLS: usize = 5;

pub type LoadDoublewordCoreAir = LoadCoreAir<
    LOAD_WIDTH_DOUBLEWORD,
    LOAD_DOUBLEWORD_SELECTOR_WIDTH,
    LOAD_DOUBLEWORD_OVERLAP_CELLS,
>;
pub type LoadDoublewordFiller = LoadFiller<
    Rv64LoadMultiByteAdapterFiller,
    LOAD_WIDTH_DOUBLEWORD,
    LOAD_DOUBLEWORD_SELECTOR_WIDTH,
    LOAD_DOUBLEWORD_OVERLAP_CELLS,
>;

pub type Rv64LoadDoublewordAir = VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadDoublewordCoreAir>;
pub type Rv64LoadDoublewordExecutor =
    LoadExecutor<Rv64LoadMultiByteAdapterExecutor<LOAD_WIDTH_DOUBLEWORD>, LOAD_WIDTH_DOUBLEWORD>;
pub type Rv64LoadDoublewordChip<F> = VmChipWrapper<F, LoadDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
