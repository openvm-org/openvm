use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{Rv32MultAdapterAir, Rv32MultAdapterStep};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32MulHAir =
    VmAirWrapper<Rv32MultAdapterAir, MulHCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32MulHStep = MulHStep<Rv32MultAdapterStep, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32MulHChip<F> = NewVmChipWrapper<F, Rv32MulHAir, Rv32MulHStep>;
