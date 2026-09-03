use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{BaseAluRegU16AdapterAir, U16_BITS};

mod core;
mod execution;
pub(crate) mod trace;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

// u16 core (4 limbs of 16 bits), shared with shift_w (SRAW) and bigint ShiftRightArithmetic256.
pub type ShiftRightArithmeticAir =
    VmAirWrapper<BaseAluRegU16AdapterAir, ShiftRightArithmeticCoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type ShiftRightArithmeticExecutor = ShiftRightArithmeticCoreExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type ShiftRightArithmeticChip<F> = VmChipWrapper<F, ShiftRightArithmeticFiller>;
