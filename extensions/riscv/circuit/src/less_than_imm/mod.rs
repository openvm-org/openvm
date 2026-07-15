use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor, Rv64BaseAluImmU16AdapterFiller,
    U16_BITS,
};

mod core;
mod execution;
pub use core::*;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

// Immediate-only variant of the less_than chip (SLTI/SLTIU): single-read immediate adapter plus
// a core with a compact signed 12-bit immediate representation.
pub type Rv64LessThanImmAir =
    VmAirWrapper<Rv64BaseAluImmU16AdapterAir, LessThanImmCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64LessThanImmExecutor =
    LessThanImmExecutor<Rv64BaseAluImmU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64LessThanImmChip<F> =
    VmChipWrapper<F, LessThanImmFiller<Rv64BaseAluImmU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>>;
