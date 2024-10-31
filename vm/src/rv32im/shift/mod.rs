use super::adapters::{
    Rv32HeapAdapterChip, INT256_NUM_LIMBS, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32BaseAluAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32ShiftChip<F> = VmChipWrapper<
    F,
    Rv32BaseAluAdapterChip<F>,
    ShiftCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Rv32Shift256Chip<F> = VmChipWrapper<
    F,
    Rv32HeapAdapterChip<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    ShiftCoreChip<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
