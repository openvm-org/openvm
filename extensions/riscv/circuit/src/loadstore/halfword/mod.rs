use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor},
    loadstore::{
        aligned::{LoadStoreAlignedCoreAir, LoadStoreAlignedFiller},
        common::{LoadStoreExecutor, KIND_HALFWORD},
    },
};

pub const HALFWORD_CASES: usize = 8;
pub const HALFWORD_SELECTOR_WIDTH: usize = 3;

pub type LoadStoreHalfwordCoreAir =
    LoadStoreAlignedCoreAir<KIND_HALFWORD, HALFWORD_CASES, HALFWORD_SELECTOR_WIDTH>;
pub type LoadStoreHalfwordFiller = LoadStoreAlignedFiller<
    crate::adapters::Rv64LoadStoreAdapterFiller,
    KIND_HALFWORD,
    HALFWORD_CASES,
    HALFWORD_SELECTOR_WIDTH,
>;

pub type Rv64LoadStoreHalfwordAir = VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreHalfwordCoreAir>;
pub type Rv64LoadStoreHalfwordExecutor =
    LoadStoreExecutor<Rv64LoadStoreAdapterExecutor, KIND_HALFWORD>;
pub type Rv64LoadStoreHalfwordChip<F> = VmChipWrapper<F, LoadStoreHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
