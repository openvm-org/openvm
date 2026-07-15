use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor, Rv64BaseAluImmU16AdapterFiller,
    Rv64BaseAluWImmU16AdapterAir, Rv64BaseAluWImmU16AdapterExecutor,
    Rv64BaseAluWImmU16AdapterFiller, RV64_WORD_U16_LIMBS, U16_BITS,
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

// Immediate-only variant of the shift_logical chip (SLLI/SRLI): single-read immediate adapter
// plus a core with no `c` limbs because the shift markers encode the amount.
pub type Rv64ShiftLogicalImmAir =
    VmAirWrapper<Rv64BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64ShiftLogicalImmExecutor =
    ShiftLogicalImmExecutor<Rv64BaseAluImmU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64ShiftLogicalImmChip<F> = VmChipWrapper<
    F,
    ShiftLogicalImmFiller<Rv64BaseAluImmU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>,
>;

pub type Rv64ShiftWLogicalImmAir = VmAirWrapper<
    Rv64BaseAluWImmU16AdapterAir,
    ShiftLogicalImmCoreAir<RV64_WORD_U16_LIMBS, U16_BITS>,
>;
pub type Rv64ShiftWLogicalImmExecutor =
    ShiftLogicalImmExecutor<Rv64BaseAluWImmU16AdapterExecutor, RV64_WORD_U16_LIMBS, U16_BITS>;
pub type Rv64ShiftWLogicalImmChip<F> = VmChipWrapper<
    F,
    ShiftLogicalImmFiller<Rv64BaseAluWImmU16AdapterFiller, RV64_WORD_U16_LIMBS, U16_BITS>,
>;
