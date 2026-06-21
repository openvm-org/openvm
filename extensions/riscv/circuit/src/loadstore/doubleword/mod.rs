use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor},
    loadstore::common::{LoadStoreExecutor, KIND_DOUBLEWORD},
};

mod core;
pub use core::*;

pub type Rv64LoadStoreDoublewordAir =
    VmAirWrapper<Rv64LoadStoreAdapterAir, LoadStoreDoublewordCoreAir>;
pub type Rv64LoadStoreDoublewordExecutor =
    LoadStoreExecutor<Rv64LoadStoreAdapterExecutor, KIND_DOUBLEWORD>;
pub type Rv64LoadStoreDoublewordChip<F> = VmChipWrapper<F, LoadStoreDoublewordFiller>;
