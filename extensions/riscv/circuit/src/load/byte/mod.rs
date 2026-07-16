use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadByteAdapterAir, Rv64LoadByteAdapterExecutor, BYTE_ACCESS_WIDTH},
    load::common::LoadExecutor,
};

mod core;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64LoadByteAir = VmAirWrapper<Rv64LoadByteAdapterAir, LoadByteCoreAir>;
pub type Rv64LoadByteExecutor = LoadExecutor<Rv64LoadByteAdapterExecutor, BYTE_ACCESS_WIDTH, 1>;
pub type Rv64LoadByteChip<F> = VmChipWrapper<F, LoadByteFiller>;
