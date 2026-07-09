use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, LOAD_WIDTH_BYTE},
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

pub type Rv64LoadByteAir = VmAirWrapper<Rv64LoadAdapterAir, LoadByteCoreAir>;
pub type Rv64LoadByteExecutor = LoadExecutor<Rv64LoadAdapterExecutor, LOAD_WIDTH_BYTE>;
pub type Rv64LoadByteChip<F> = VmChipWrapper<F, LoadByteFiller>;
