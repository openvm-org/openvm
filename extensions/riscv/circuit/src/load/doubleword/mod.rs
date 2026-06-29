use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller},
    load::{
        common::{LoadExecutor, KIND_DOUBLEWORD},
        width_aligned::{LoadWidthAlignedCoreAir, LoadWidthAlignedFiller},
    },
};

pub const DOUBLEWORD_LOAD_CASES: usize = 1;
pub const DOUBLEWORD_LOAD_SELECTOR_WIDTH: usize = 1;

pub type LoadDoublewordCoreAir =
    LoadWidthAlignedCoreAir<KIND_DOUBLEWORD, DOUBLEWORD_LOAD_CASES, DOUBLEWORD_LOAD_SELECTOR_WIDTH>;
pub type LoadDoublewordFiller = LoadWidthAlignedFiller<
    Rv64LoadAdapterFiller,
    KIND_DOUBLEWORD,
    DOUBLEWORD_LOAD_CASES,
    DOUBLEWORD_LOAD_SELECTOR_WIDTH,
>;

pub type Rv64LoadDoublewordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadDoublewordCoreAir>;
pub type Rv64LoadDoublewordExecutor = LoadExecutor<Rv64LoadAdapterExecutor, KIND_DOUBLEWORD>;
pub type Rv64LoadDoublewordChip<F> = VmChipWrapper<F, LoadDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
