use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use super::adapters::{
    Rv32BaseAluAdapterAir, Rv32BaseAluAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use crate::adapters::Rv32BaseAluAdapterChip;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BaseAluAir =
    VmAirWrapper<Rv32BaseAluAdapterAir, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv32BaseAluStep =
    BaseAluStep<Rv32BaseAluAdapterStep<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32BaseAluChip<F> =
    BaseAluChip<F, Rv32BaseAluAdapterChip<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
