use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadByteAdapterAir, BYTE_ACCESS_WIDTH},
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

pub type Rv64LoadByteAir = VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir>;
pub type Rv64LoadByteExecutor = LoadExecutor<BYTE_ACCESS_WIDTH>;
pub type Rv64LoadByteChip<F> = VmChipWrapper<F, LoadByteFiller>;
