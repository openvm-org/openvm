use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64BaseAluWAdapterAir, Rv64BaseAluWAdapterExecutor, Rv64BaseAluWAdapterFiller,
};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64ShiftWAir = VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftWCoreAir>;
pub type Rv64ShiftWExecutor = ShiftWExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64ShiftWChip<F> = VmChipWrapper<F, ShiftWFiller<Rv64BaseAluWAdapterFiller>>;
