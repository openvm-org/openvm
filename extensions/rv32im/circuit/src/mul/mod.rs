use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32MultAdapterAir, Rv32MultAdapterFiller, Rv32MultAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32MultiplicationAir = VmAirWrapper<
    Rv32MultAdapterAir,
    MultiplicationCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32MultiplicationStep =
    MultiplicationStep<Rv32MultAdapterStep, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32MultiplicationChip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<Rv32MultAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
