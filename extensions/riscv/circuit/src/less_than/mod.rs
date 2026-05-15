use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BaseAluU16AdapterAir, Rv64BaseAluU16AdapterExecutor, Rv64BaseAluU16AdapterFiller,
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

/// RV64 less_than reads/writes 4 u16 cells per register and compares with
/// `LIMB_BITS = 16` per-limb range checks.
pub const RV64_LESS_THAN_NUM_LIMBS: usize = BLOCK_FE_WIDTH;
pub const RV64_LESS_THAN_LIMB_BITS: usize = 16;

pub type Rv64LessThanAir = VmAirWrapper<
    Rv64BaseAluU16AdapterAir,
    LessThanCoreAir<RV64_LESS_THAN_NUM_LIMBS, RV64_LESS_THAN_LIMB_BITS>,
>;
pub type Rv64LessThanExecutor = LessThanExecutor<
    Rv64BaseAluU16AdapterExecutor,
    RV64_LESS_THAN_NUM_LIMBS,
    RV64_LESS_THAN_LIMB_BITS,
>;
pub type Rv64LessThanChip<F> = VmChipWrapper<
    F,
    LessThanFiller<Rv64BaseAluU16AdapterFiller, RV64_LESS_THAN_NUM_LIMBS, RV64_LESS_THAN_LIMB_BITS>,
>;
