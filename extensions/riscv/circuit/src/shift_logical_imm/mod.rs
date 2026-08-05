use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    BaseAluImmU16AdapterAir, BaseAluWImmU16AdapterAir, U16_BITS, WORD_U16_LIMBS,
};

mod core;
mod execution;
pub use core::*;

pub(crate) mod trace;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

// Immediate-only variant of the shift_logical chip (SLLI/SRLI): single-read immediate adapter
// plus a core with no `c` limbs because the shift markers encode the amount.
pub type ShiftLogicalImmAir =
    VmAirWrapper<BaseAluImmU16AdapterAir, ShiftLogicalImmCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type ShiftLogicalImmExecutor = ShiftLogicalImmCoreExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type ShiftLogicalImmChip<F> = VmChipWrapper<F, ShiftLogicalImmFiller>;

pub type ShiftWLogicalImmAir =
    VmAirWrapper<BaseAluWImmU16AdapterAir, ShiftLogicalImmCoreAir<WORD_U16_LIMBS, U16_BITS>>;
pub type ShiftWLogicalImmExecutor = ShiftLogicalImmCoreExecutor<WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWLogicalImmChip<F> = VmChipWrapper<F, ShiftLogicalImmFiller>;
