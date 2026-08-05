use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{LoadByteAdapterAir, BYTE_ACCESS_WIDTH},
    load::common::LoadExecutor,
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

pub type LoadByteAir = VmAirWrapper<LoadByteAdapterAir, LoadByteCoreAir>;
pub type LoadByteExecutor = LoadExecutor<BYTE_ACCESS_WIDTH>;
pub type LoadByteChip<F> = VmChipWrapper<F, LoadByteFiller>;
