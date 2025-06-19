use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32MultAdapterAir, Rv32MultAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32MultiplicationChip = MultiplicationStep<
    Rv32MultiplicationAir,
    Rv32MultAdapterStep,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
