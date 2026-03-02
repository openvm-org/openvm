use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller, RV64_CELL_BITS,
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

pub type Rv64BaseAluWAir = VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluWCoreAir>;
pub type Rv64BaseAluWExecutor = BaseAluWExecutor<Rv64BaseAluAdapterExecutor<RV64_CELL_BITS>>;
pub type Rv64BaseAluWChip<F> =
    VmChipWrapper<F, BaseAluWFiller<Rv64BaseAluAdapterFiller<RV64_CELL_BITS>>>;
