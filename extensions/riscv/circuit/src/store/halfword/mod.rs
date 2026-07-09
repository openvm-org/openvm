use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller, STORE_WIDTH_HALFWORD,
    },
    store::{
        common::StoreExecutor,
        width_aligned::{StoreWidthAlignedCoreAir, StoreWidthAlignedFiller},
    },
};

pub const HALFWORD_STORE_CASES: usize = 4;
pub const HALFWORD_STORE_SELECTOR_WIDTH: usize = 2;

pub type StoreHalfwordCoreAir = StoreWidthAlignedCoreAir<
    STORE_WIDTH_HALFWORD,
    HALFWORD_STORE_CASES,
    HALFWORD_STORE_SELECTOR_WIDTH,
>;
pub type StoreHalfwordFiller = StoreWidthAlignedFiller<
    Rv64StoreAdapterFiller,
    STORE_WIDTH_HALFWORD,
    HALFWORD_STORE_CASES,
    HALFWORD_STORE_SELECTOR_WIDTH,
>;

pub type Rv64StoreHalfwordAir = VmAirWrapper<Rv64StoreAdapterAir, StoreHalfwordCoreAir>;
pub type Rv64StoreHalfwordExecutor = StoreExecutor<Rv64StoreAdapterExecutor, STORE_WIDTH_HALFWORD>;
pub type Rv64StoreHalfwordChip<F> = VmChipWrapper<F, StoreHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
