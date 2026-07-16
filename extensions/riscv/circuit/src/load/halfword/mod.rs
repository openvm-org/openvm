use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH,
    },
    load::{
        common::LoadExecutor,
        core::{LoadCoreAir, LoadFiller},
    },
};

/// Cells overlapped by an odd-shift halfword load: `HALFWORD_ACCESS_WIDTH / 2 + 1`.
pub const LOAD_HALFWORD_OVERLAP_CELLS: usize = HALFWORD_ACCESS_WIDTH / 2 + 1;

pub type LoadHalfwordCoreAir = LoadCoreAir<HALFWORD_ACCESS_WIDTH, LOAD_HALFWORD_OVERLAP_CELLS>;
pub type LoadHalfwordFiller =
    LoadFiller<Rv64LoadMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH, LOAD_HALFWORD_OVERLAP_CELLS>;

pub type Rv64LoadHalfwordAir = VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadHalfwordCoreAir>;
pub type Rv64LoadHalfwordExecutor =
    LoadExecutor<Rv64LoadMultiByteAdapterExecutor<HALFWORD_ACCESS_WIDTH>, HALFWORD_ACCESS_WIDTH>;
pub type Rv64LoadHalfwordChip<F> = VmChipWrapper<F, LoadHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
