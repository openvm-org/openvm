use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32LoadSignExtendChip = LoadSignExtendStep<
    Rv32LoadSignExtendAir,
    Rv32LoadStoreAdapterStep,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
