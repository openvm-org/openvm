use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::{
    adapters::{
        Rv64BaseAluU16ImmAdapterAir, Rv64BaseAluU16ImmAdapterExecutor,
        Rv64BaseAluU16ImmAdapterFiller, U16_BITS,
    },
    ShiftRightArithmeticCoreAir, ShiftRightArithmeticExecutor, ShiftRightArithmeticFiller,
};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

// Immediate-only variant of the shift_right_arithmetic chip (SRAI).
pub type Rv64ShiftRightArithmeticImmAir = VmAirWrapper<
    Rv64BaseAluU16ImmAdapterAir,
    ShiftRightArithmeticCoreAir<BLOCK_FE_WIDTH, U16_BITS>,
>;
pub type Rv64ShiftRightArithmeticImmExecutor =
    ShiftRightArithmeticExecutor<Rv64BaseAluU16ImmAdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64ShiftRightArithmeticImmChip<F> = VmChipWrapper<
    F,
    ShiftRightArithmeticFiller<Rv64BaseAluU16ImmAdapterFiller, BLOCK_FE_WIDTH, U16_BITS>,
>;
