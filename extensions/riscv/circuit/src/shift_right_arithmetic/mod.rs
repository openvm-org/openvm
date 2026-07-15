use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64BaseAluRegU16AdapterAir, Rv64BaseAluRegU16AdapterExecutor, Rv64BaseAluRegU16AdapterFiller,
    U16_BITS,
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
pub type Rv64ShiftRightArithmeticAir = VmAirWrapper<
    Rv64BaseAluRegU16AdapterAir,
    ShiftRightArithmeticCoreAir<BLOCK_FE_WIDTH, U16_BITS>,
>;
pub type Rv64ShiftRightArithmeticExecutor =
    ShiftRightArithmeticExecutor<Rv64BaseAluRegU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64ShiftRightArithmeticChip<F> = VmChipWrapper<
    F,
    ShiftRightArithmeticFiller<Rv64BaseAluRegU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>,
>;
