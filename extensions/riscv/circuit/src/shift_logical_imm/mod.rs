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

// Immediate-only variant of the shift_logical chip (SLLI/SRLI): single-read immediate adapter
// plus a forked core with no `c` limbs (the shift markers already encode the amount).
pub type Rv64ShiftLogicalImmAir =
    VmAirWrapper<Rv64ImmBaseAluU16AdapterAir, ShiftLogicalImmCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64ShiftLogicalImmExecutor =
    ShiftLogicalImmExecutor<Rv64ImmBaseAluU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64ShiftLogicalImmChip<F> = VmChipWrapper<
    F,
    ShiftLogicalImmFiller<Rv64ImmBaseAluU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>,
>;
