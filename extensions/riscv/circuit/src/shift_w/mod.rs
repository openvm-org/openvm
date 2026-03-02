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

pub type Rv64ShiftWAir = VmAirWrapper<Rv64BaseAluAdapterAir, ShiftWCoreAir>;
pub type Rv64ShiftWExecutor = ShiftWExecutor<Rv64BaseAluAdapterExecutor<RV64_CELL_BITS>>;
pub type Rv64ShiftWChip<F> =
    VmChipWrapper<F, ShiftWFiller<Rv64BaseAluAdapterFiller<RV64_CELL_BITS>>>;
