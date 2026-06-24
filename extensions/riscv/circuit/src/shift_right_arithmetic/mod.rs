use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64BaseAluU16AdapterAir, Rv64BaseAluU16AdapterExecutor, Rv64BaseAluU16AdapterFiller, U16_BITS,
};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

// u16 core (4 limbs of 16 bits), shared with shift_w (SRAW) and bigint ShiftRightArithmetic256.
pub type Rv64ShiftRightArithmeticAir =
    VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftRightArithmeticCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64ShiftRightArithmeticExecutor =
    ShiftRightArithmeticExecutor<Rv64BaseAluU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64ShiftRightArithmeticChip<F> = VmChipWrapper<
    F,
    ShiftRightArithmeticFiller<Rv64BaseAluU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>,
>;
