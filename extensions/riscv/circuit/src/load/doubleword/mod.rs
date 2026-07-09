use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, LOAD_WIDTH_DOUBLEWORD,
    },
    load::{
        common::LoadExecutor,
        core::{LoadWidthAlignedCoreAir, LoadWidthAlignedFiller},
    },
};

pub const LOAD_DOUBLEWORD_SELECTOR_WIDTH: usize = 1;

pub type LoadDoublewordCoreAir =
    LoadWidthAlignedCoreAir<LOAD_WIDTH_DOUBLEWORD, LOAD_DOUBLEWORD_SELECTOR_WIDTH>;
pub type LoadDoublewordFiller = LoadWidthAlignedFiller<
    Rv64LoadAdapterFiller,
    LOAD_WIDTH_DOUBLEWORD,
    LOAD_DOUBLEWORD_SELECTOR_WIDTH,
>;

pub type Rv64LoadDoublewordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadDoublewordCoreAir>;
pub type Rv64LoadDoublewordExecutor = LoadExecutor<Rv64LoadAdapterExecutor, LOAD_WIDTH_DOUBLEWORD>;
pub type Rv64LoadDoublewordChip<F> = VmChipWrapper<F, LoadDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
