mod core;

pub use core::*;

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterStep};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreChip =
    LoadStoreStep<Rv32LoadStoreAir, Rv32LoadStoreAdapterStep, RV32_REGISTER_NUM_LIMBS>;
