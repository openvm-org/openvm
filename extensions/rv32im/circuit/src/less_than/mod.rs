use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use super::adapters::{
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32LessThanAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, LessThanCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32LessThanStep =
    LessThanStep<Rv32BaseAluAdapterStep<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32LessThanChip<F> =
    NewVmChipWrapper<F, Rv32LessThanAir, Rv32LessThanStep, MatrixRecordArena<F>>;
