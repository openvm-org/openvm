use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32MultAdapterAir, Rv32MultAdapterExecutor, Rv32MultAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv32MulHAir =
    VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32MulHExecutor =
    MulHExecutor<Rv32MultAdapterExecutor, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32MulHChip<F> =
    VmChipWrapper<F, MulHFiller<Rv32MultAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
