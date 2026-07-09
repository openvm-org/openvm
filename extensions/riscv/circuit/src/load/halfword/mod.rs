use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, LOAD_WIDTH_HALFWORD,
    },
    load::{
        common::LoadExecutor,
        core::{LoadWidthAlignedCoreAir, LoadWidthAlignedFiller},
    },
};

pub const HALFWORD_LOAD_CASES: usize = 4;
pub const HALFWORD_LOAD_SELECTOR_WIDTH: usize = 2;

pub type LoadHalfwordCoreAir =
    LoadWidthAlignedCoreAir<LOAD_WIDTH_HALFWORD, HALFWORD_LOAD_CASES, HALFWORD_LOAD_SELECTOR_WIDTH>;
pub type LoadHalfwordFiller = LoadWidthAlignedFiller<
    Rv64LoadAdapterFiller,
    LOAD_WIDTH_HALFWORD,
    HALFWORD_LOAD_CASES,
    HALFWORD_LOAD_SELECTOR_WIDTH,
>;

pub type Rv64LoadHalfwordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadHalfwordCoreAir>;
pub type Rv64LoadHalfwordExecutor = LoadExecutor<Rv64LoadAdapterExecutor, LOAD_WIDTH_HALFWORD>;
pub type Rv64LoadHalfwordChip<F> = VmChipWrapper<F, LoadHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
