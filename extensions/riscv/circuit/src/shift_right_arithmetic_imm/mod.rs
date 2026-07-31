use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BaseAluImmU16AdapterAir, Rv64BaseAluWImmU16AdapterAir, RV64_WORD_U16_LIMBS, U16_BITS,
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
pub type Rv64ShiftRightArithmeticImmAir = VmAirWrapper<
    Rv64BaseAluImmU16AdapterAir,
    ShiftRightArithmeticImmCoreAir<BLOCK_FE_WIDTH, U16_BITS>,
>;
pub type Rv64ShiftRightArithmeticImmExecutor =
    ShiftRightArithmeticImmExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64ShiftRightArithmeticImmChip<F> = VmChipWrapper<F, ShiftRightArithmeticImmFiller>;

pub type Rv64ShiftWRightArithmeticImmAir = VmAirWrapper<
    Rv64BaseAluWImmU16AdapterAir,
    ShiftRightArithmeticImmCoreAir<RV64_WORD_U16_LIMBS, U16_BITS>,
>;
pub type Rv64ShiftWRightArithmeticImmExecutor =
    ShiftRightArithmeticImmExecutor<RV64_WORD_U16_LIMBS, U16_BITS>;
pub type Rv64ShiftWRightArithmeticImmChip<F> = VmChipWrapper<F, ShiftRightArithmeticImmFiller>;
