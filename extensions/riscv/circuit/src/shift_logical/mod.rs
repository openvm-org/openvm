use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64BaseAluU16AdapterAir, Rv64BaseAluU16AdapterExecutor, Rv64BaseAluU16AdapterFiller, U16_BITS,
};

mod core;
mod core_u16;
mod execution;
pub use core::*;

pub use core_u16::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

// The rv64 register shift uses u16 limbs (4 limbs of 16 bits). The byte-shaped `ShiftLogicalCore*`
// types above remain for shift_w (SLLW/SRLW) and bigint Shift256, whose AIRs rely on byte-level
// carry/shift structure.
pub type Rv64ShiftLogicalAir =
    VmAirWrapper<Rv64BaseAluU16AdapterAir, ShiftLogicalU16CoreAir<BLOCK_FE_WIDTH, U16_BITS>>;
pub type Rv64ShiftLogicalExecutor =
    ShiftLogicalU16Executor<Rv64BaseAluU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64ShiftLogicalChip<F> =
    VmChipWrapper<F, ShiftLogicalU16Filler<Rv64BaseAluU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS>>;
