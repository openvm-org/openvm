use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
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
