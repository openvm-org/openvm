use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller},
    store::{
        aligned::{StoreAlignedCoreAir, StoreAlignedFiller},
        common::{StoreExecutor, KIND_HALFWORD},
    },
};

pub const HALFWORD_STORE_CASES: usize = 4;
pub const HALFWORD_STORE_SELECTOR_WIDTH: usize = 2;

pub type StoreHalfwordCoreAir =
    StoreAlignedCoreAir<KIND_HALFWORD, HALFWORD_STORE_CASES, HALFWORD_STORE_SELECTOR_WIDTH>;
pub type StoreHalfwordFiller = StoreAlignedFiller<
    Rv64StoreAdapterFiller,
    KIND_HALFWORD,
    HALFWORD_STORE_CASES,
    HALFWORD_STORE_SELECTOR_WIDTH,
>;

pub type Rv64StoreHalfwordAir = VmAirWrapper<Rv64StoreAdapterAir, StoreHalfwordCoreAir>;
pub type Rv64StoreHalfwordExecutor = StoreExecutor<Rv64StoreAdapterExecutor, KIND_HALFWORD>;
pub type Rv64StoreHalfwordChip<F> = VmChipWrapper<F, StoreHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
