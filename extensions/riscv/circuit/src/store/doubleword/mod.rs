use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64StoreAdapterAir, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller,
        STORE_WIDTH_DOUBLEWORD,
    },
    store::{
        common::StoreExecutor,
        core::{StoreCoreAir, StoreFiller},
    },
};

pub const STORE_DOUBLEWORD_SELECTOR_WIDTH: usize = 1;

pub type StoreDoublewordCoreAir =
    StoreCoreAir<STORE_WIDTH_DOUBLEWORD, STORE_DOUBLEWORD_SELECTOR_WIDTH>;
pub type StoreDoublewordFiller =
    StoreFiller<Rv64StoreAdapterFiller, STORE_WIDTH_DOUBLEWORD, STORE_DOUBLEWORD_SELECTOR_WIDTH>;

pub type Rv64StoreDoublewordAir = VmAirWrapper<Rv64StoreAdapterAir, StoreDoublewordCoreAir>;
pub type Rv64StoreDoublewordExecutor =
    StoreExecutor<Rv64StoreAdapterExecutor, STORE_WIDTH_DOUBLEWORD>;
pub type Rv64StoreDoublewordChip<F> = VmChipWrapper<F, StoreDoublewordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
