use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller},
    load::{
        aligned::{LoadAlignedCoreAir, LoadAlignedFiller},
        common::{LoadExecutor, KIND_HALFWORD},
    },
};

pub const HALFWORD_LOAD_CASES: usize = 4;
pub const HALFWORD_LOAD_SELECTOR_WIDTH: usize = 2;

pub type LoadHalfwordCoreAir =
    LoadAlignedCoreAir<KIND_HALFWORD, HALFWORD_LOAD_CASES, HALFWORD_LOAD_SELECTOR_WIDTH>;
pub type LoadHalfwordFiller = LoadAlignedFiller<
    Rv64LoadAdapterFiller,
    KIND_HALFWORD,
    HALFWORD_LOAD_CASES,
    HALFWORD_LOAD_SELECTOR_WIDTH,
>;

pub type Rv64LoadHalfwordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadHalfwordCoreAir>;
pub type Rv64LoadHalfwordExecutor = LoadExecutor<Rv64LoadAdapterExecutor, KIND_HALFWORD>;
pub type Rv64LoadHalfwordChip<F> = VmChipWrapper<F, LoadHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
