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

pub type Rv64BaseAluWAir = VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluWCoreAir>;
pub type Rv64BaseAluWExecutor = BaseAluWExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64BaseAluWChip<F> = VmChipWrapper<F, BaseAluWFiller<Rv64BaseAluWAdapterFiller>>;
