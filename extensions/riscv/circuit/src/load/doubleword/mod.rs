use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, LOAD_WIDTH_DOUBLEWORD,
    },
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

pub const LOAD_DOUBLEWORD_SELECTOR_WIDTH: usize = 3;
/// Cells loaded by a doubleword load: `LOAD_WIDTH_DOUBLEWORD / 2`.
pub const LOAD_DOUBLEWORD_LOADED_CELLS: usize = 4;

pub type LoadDoublewordCoreAir = LoadCoreAir<
    LOAD_WIDTH_DOUBLEWORD,
    LOAD_DOUBLEWORD_SELECTOR_WIDTH,
    LOAD_DOUBLEWORD_LOADED_CELLS,
>;
pub type LoadDoublewordFiller = LoadFiller<
    Rv64LoadAdapterFiller,
    LOAD_WIDTH_DOUBLEWORD,
    LOAD_DOUBLEWORD_SELECTOR_WIDTH,
    LOAD_DOUBLEWORD_LOADED_CELLS,
>;

pub type Rv64LoadDoublewordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadDoublewordCoreAir>;
pub type Rv64LoadDoublewordExecutor =
    LoadExecutor<Rv64LoadAdapterExecutor<LOAD_WIDTH_DOUBLEWORD>, LOAD_WIDTH_DOUBLEWORD>;
pub type Rv64LoadDoublewordChip<F> = VmChipWrapper<F, LoadDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
