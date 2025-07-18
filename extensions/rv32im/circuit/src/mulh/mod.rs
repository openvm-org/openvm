use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32MultAdapterAir, Rv32MultAdapterFiller, Rv32MultAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32MulHAir =
    VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32MulHStep = MulHStep<Rv32MultAdapterStep, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32MulHChip<F> =
    VmChipWrapper<F, MulHFiller<Rv32MultAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
