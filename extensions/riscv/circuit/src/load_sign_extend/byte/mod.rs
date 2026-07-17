use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadByteAdapterAir, Rv64LoadByteAdapterExecutor, BYTE_ACCESS_WIDTH},
    load_sign_extend::common::LoadSignExtendExecutor,
};

mod core;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64LoadSignExtendByteAir =
    VmAirWrapper<Rv64LoadByteAdapterAir, LoadSignExtendByteCoreAir>;
pub type Rv64LoadSignExtendByteExecutor =
    LoadSignExtendExecutor<Rv64LoadByteAdapterExecutor, BYTE_ACCESS_WIDTH, 1>;
pub type Rv64LoadSignExtendByteChip<F> = VmChipWrapper<F, LoadSignExtendByteFiller>;
