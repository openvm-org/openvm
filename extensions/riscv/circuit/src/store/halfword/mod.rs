use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller, STORE_WIDTH_HALFWORD,
    },
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

pub const STORE_HALFWORD_SELECTOR_WIDTH: usize = 3;
/// Source register cells decomposed on an odd-shift halfword store: `STORE_WIDTH_HALFWORD / 2`.
pub const STORE_HALFWORD_VALUE_CELLS: usize = 1;

pub type StoreHalfwordCoreAir =
    StoreCoreAir<STORE_WIDTH_HALFWORD, STORE_HALFWORD_SELECTOR_WIDTH, STORE_HALFWORD_VALUE_CELLS>;
pub type StoreHalfwordFiller = StoreFiller<
    Rv64StoreAdapterFiller,
    STORE_WIDTH_HALFWORD,
    STORE_HALFWORD_SELECTOR_WIDTH,
    STORE_HALFWORD_VALUE_CELLS,
>;

pub type Rv64StoreHalfwordAir = VmAirWrapper<Rv64StoreAdapterAir, StoreHalfwordCoreAir>;
pub type Rv64StoreHalfwordExecutor =
    StoreExecutor<Rv64StoreAdapterExecutor<STORE_WIDTH_HALFWORD>, STORE_WIDTH_HALFWORD>;
pub type Rv64StoreHalfwordChip<F> = VmChipWrapper<F, StoreHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
