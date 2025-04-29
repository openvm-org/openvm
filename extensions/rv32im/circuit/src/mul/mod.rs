use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{Rv32MultAdapterAir, Rv32MultAdapterStep};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

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
pub type Rv32MultiplicationChip<F> =
    NewVmChipWrapper<F, Rv32MultiplicationAir, Rv32MultiplicationStep>;
