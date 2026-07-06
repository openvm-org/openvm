use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64ImmBaseAluU16AdapterAir, Rv64ImmBaseAluU16AdapterExecutor, Rv64ImmBaseAluU16AdapterFiller,
    U16_BITS,
};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

// Immediate-only variant of the less_than chip (SLTI/SLTIU): single-read immediate adapter plus
// a forked core with the ADDI-style two-column immediate instead of the `c` limbs.
pub type Rv64LessThanImmAir =
    VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, LessThanImmCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64LessThanImmExecutor =
    LessThanImmExecutor<Rv64ImmBaseAluU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64LessThanImmChip<F> =
    VmChipWrapper<F, LessThanImmFiller<Rv64ImmBaseAluU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>>;
