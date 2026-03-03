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

pub type Rv32DivRemAir =
    VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32DivRemExecutor =
    DivRemExecutor<Rv32MultAdapterExecutor, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32DivRemChip<F> =
    VmChipWrapper<F, DivRemFiller<Rv32MultAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
