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

// Immediate-only variant of the shift_right_arithmetic chip (SRAI): single-read immediate
// adapter plus a core which reconstructs the immediate from its shift markers.
pub type Rv64ShiftRightArithmeticImmAir = VmAirWrapper<
    Rv64BaseAluImmU16AdapterAir,
    ShiftRightArithmeticImmCoreAir<BLOCK_FE_WIDTH, U16_BITS>,
>;
pub type Rv64ShiftRightArithmeticImmExecutor =
    ShiftRightArithmeticImmExecutor<Rv64BaseAluImmU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64ShiftRightArithmeticImmChip<F> = VmChipWrapper<
    F,
    ShiftRightArithmeticImmFiller<Rv64BaseAluImmU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>,
>;

pub type Rv64ShiftWRightArithmeticImmAir = VmAirWrapper<
    Rv64BaseAluWImmU16AdapterAir,
    ShiftRightArithmeticImmCoreAir<RV64_WORD_U16_LIMBS, U16_BITS>,
>;
pub type Rv64ShiftWRightArithmeticImmExecutor = ShiftRightArithmeticImmExecutor<
    Rv64BaseAluWImmU16AdapterExecutor,
    RV64_WORD_U16_LIMBS,
    U16_BITS,
>;
pub type Rv64ShiftWRightArithmeticImmChip<F> = VmChipWrapper<
    F,
    ShiftRightArithmeticImmFiller<Rv64BaseAluWImmU16AdapterFiller, RV64_WORD_U16_LIMBS, U16_BITS>,
>;
