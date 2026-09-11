use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{LoadByteAdapterAir, BYTE_ACCESS_WIDTH},
    load_sign_extend::common::LoadSignExtendExecutor,
};

mod core;
pub use core::*;

pub(crate) mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type LoadSignExtendByteAir = VmAirWrapper<LoadByteAdapterAir, LoadSignExtendByteCoreAir>;
pub type LoadSignExtendByteExecutor = LoadSignExtendExecutor<BYTE_ACCESS_WIDTH>;
pub type LoadSignExtendByteChip<F> = VmChipWrapper<F, LoadSignExtendByteFiller>;
