use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32MultAdapterAir, Rv32MultAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32DivRemAir =
    VmAirWrapper<Rv32MultAdapterAir, DivRemCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32DivRemStep = DivRemStep<Rv32MultAdapterStep, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32DivRemChip<F> =
    NewVmChipWrapper<F, Rv32DivRemAir, Rv32DivRemStep, MatrixRecordArena<F>>;
