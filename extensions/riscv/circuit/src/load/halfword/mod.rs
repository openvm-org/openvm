use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, LOAD_WIDTH_HALFWORD,
    },
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

pub const LOAD_HALFWORD_SELECTOR_WIDTH: usize = 3;
/// Cells overlapped by an odd-shift halfword load: `LOAD_WIDTH_HALFWORD / 2 + 1`.
pub const LOAD_HALFWORD_OVERLAP_CELLS: usize = 2;

pub type LoadHalfwordCoreAir =
    LoadCoreAir<LOAD_WIDTH_HALFWORD, LOAD_HALFWORD_SELECTOR_WIDTH, LOAD_HALFWORD_OVERLAP_CELLS>;
pub type LoadHalfwordFiller = LoadFiller<
    Rv64LoadAdapterFiller,
    LOAD_WIDTH_HALFWORD,
    LOAD_HALFWORD_SELECTOR_WIDTH,
    LOAD_HALFWORD_OVERLAP_CELLS,
>;

pub type Rv64LoadHalfwordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadHalfwordCoreAir>;
pub type Rv64LoadHalfwordExecutor =
    LoadExecutor<Rv64LoadAdapterExecutor<LOAD_WIDTH_HALFWORD>, LOAD_WIDTH_HALFWORD>;
pub type Rv64LoadHalfwordChip<F> = VmChipWrapper<F, LoadHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
