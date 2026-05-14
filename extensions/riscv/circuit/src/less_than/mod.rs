use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BaseAluAdapterU16Air, Rv64BaseAluAdapterU16Executor, Rv64BaseAluAdapterU16Filler,
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

/// Pattern B: RV64 less_than reads/writes 4 u16 cells per register and compares with
/// `LIMB_BITS = 16` per-limb range checks.
pub const RV64_LESS_THAN_NUM_LIMBS: usize = BLOCK_FE_WIDTH;
pub const RV64_LESS_THAN_LIMB_BITS: usize = 16;

pub type Rv64LessThanAir = VmAirWrapper<
    Rv64BaseAluAdapterU16Air,
    LessThanCoreAir<RV64_LESS_THAN_NUM_LIMBS, RV64_LESS_THAN_LIMB_BITS>,
>;
pub type Rv64LessThanExecutor = LessThanExecutor<
    Rv64BaseAluAdapterU16Executor,
    RV64_LESS_THAN_NUM_LIMBS,
    RV64_LESS_THAN_LIMB_BITS,
>;
pub type Rv64LessThanChip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv64BaseAluAdapterU16Filler,
        RV64_LESS_THAN_NUM_LIMBS,
        RV64_LESS_THAN_LIMB_BITS,
    >,
>;
