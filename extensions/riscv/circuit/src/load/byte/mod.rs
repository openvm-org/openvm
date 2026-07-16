use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadByteAdapterAir, Rv64LoadByteAdapterExecutor, LOAD_WIDTH_BYTE},
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
pub type Rv64LoadByteExecutor = LoadExecutor<Rv64LoadByteAdapterExecutor, LOAD_WIDTH_BYTE, 1>;
pub type Rv64LoadByteChip<F> = VmChipWrapper<F, LoadByteFiller>;
