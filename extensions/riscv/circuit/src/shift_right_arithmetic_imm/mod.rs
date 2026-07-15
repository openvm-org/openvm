use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor, Rv64BaseAluImmU16AdapterFiller,
    U16_BITS,
};

mod core;
mod execution;
pub use core::*;

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
