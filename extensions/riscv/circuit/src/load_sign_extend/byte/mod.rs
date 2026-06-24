use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor},
    loadstore::common::{LoadStoreExecutor, KIND_BYTE},
};

mod core;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

pub type Rv64LoadSignExtendByteAir =
    VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendByteCoreAir>;
pub type Rv64LoadSignExtendByteExecutor =
    LoadStoreExecutor<Rv64LoadStoreAdapterExecutor, KIND_BYTE>;
pub type Rv64LoadSignExtendByteChip<F> = VmChipWrapper<F, LoadSignExtendByteFiller>;
