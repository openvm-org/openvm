use super::adapters::{
    Rv32VecHeapAdapterChip, RV32_CELL_BITS, RV32_INT256_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS,
};
use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32BaseAluAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BaseAluChip<F> = VmChipWrapper<
    F,
    Rv32BaseAluAdapterChip<F>,
    BaseAluCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Rv32BaseAlu256Chip<F> = VmChipWrapper<
    F,
    Rv32VecHeapAdapterChip<F, 2, 1, 1, RV32_INT256_NUM_LIMBS, RV32_INT256_NUM_LIMBS>,
    BaseAluCoreChip<RV32_INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
