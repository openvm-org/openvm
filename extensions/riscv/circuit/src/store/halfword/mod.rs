use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64StoreMultiByteAdapterAir, Rv64StoreMultiByteAdapterExecutor,
        Rv64StoreMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH,
    },
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

/// Source register cells decomposed on an odd-shift halfword store: `HALFWORD_ACCESS_WIDTH / 2`.
pub const STORE_HALFWORD_VALUE_CELLS: usize = HALFWORD_ACCESS_WIDTH / 2;

pub type StoreHalfwordCoreAir = StoreCoreAir<HALFWORD_ACCESS_WIDTH, STORE_HALFWORD_VALUE_CELLS>;
pub type StoreHalfwordFiller =
    StoreFiller<Rv64StoreMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH, STORE_HALFWORD_VALUE_CELLS>;

pub type Rv64StoreHalfwordAir = VmAirWrapper<Rv64StoreMultiByteAdapterAir, StoreHalfwordCoreAir>;
pub type Rv64StoreHalfwordExecutor =
    StoreExecutor<Rv64StoreMultiByteAdapterExecutor<HALFWORD_ACCESS_WIDTH>, HALFWORD_ACCESS_WIDTH>;
pub type Rv64StoreHalfwordChip<F> = VmChipWrapper<F, StoreHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
