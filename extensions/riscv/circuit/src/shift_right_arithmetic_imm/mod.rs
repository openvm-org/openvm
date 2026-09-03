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

// Immediate-only variant of the shift_right_arithmetic chip (SRAI): single-read immediate
// adapter plus a core which reconstructs the immediate from its shift markers.
pub type ShiftRightArithmeticImmAir =
    VmAirWrapper<BaseAluImmU16AdapterAir, ShiftRightArithmeticImmCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type ShiftRightArithmeticImmExecutor =
    ShiftRightArithmeticImmCoreExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type ShiftRightArithmeticImmChip<F> = VmChipWrapper<F, ShiftRightArithmeticImmFiller>;

pub type ShiftWRightArithmeticImmAir = VmAirWrapper<
    BaseAluWImmU16AdapterAir,
    ShiftRightArithmeticImmCoreAir<WORD_U16_LIMBS, U16_BITS>,
>;
pub type ShiftWRightArithmeticImmExecutor =
    ShiftRightArithmeticImmCoreExecutor<WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWRightArithmeticImmChip<F> = VmChipWrapper<F, ShiftRightArithmeticImmFiller>;
