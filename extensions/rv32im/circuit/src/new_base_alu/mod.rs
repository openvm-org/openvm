use openvm_circuit::arch::new_integration_api::VmChipWrapper;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::new_adapter::Rv32RegisterAdapter;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BaseAluChip<F> =
    VmChipWrapper<F, Rv32RegisterAdapter, BaseAluCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
