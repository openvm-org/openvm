use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use super::adapters::{
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32ShiftAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, ShiftCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32ShiftStep =
    ShiftStep<Rv32BaseAluAdapterStep<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32ShiftChip<F> = NewVmChipWrapper<F, Rv32ShiftAir, Rv32ShiftStep, MatrixRecordArena<F>>;
