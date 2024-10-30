use super::adapters::{
    Rv32HeapAdapterChip, RV32_CELL_BITS, RV32_INT256_NUM_LIMBS, RV32_REGISTER_NUM_LIMBS,
};
use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32MultAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32MultiplicationChip<F> = VmChipWrapper<
    F,
    Rv32MultAdapterChip<F>,
    MultiplicationCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Rv32Multiplication256Chip<F> = VmChipWrapper<
    F,
    Rv32HeapAdapterChip<F, 2, RV32_INT256_NUM_LIMBS, RV32_INT256_NUM_LIMBS>,
    MultiplicationCoreChip<RV32_INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
