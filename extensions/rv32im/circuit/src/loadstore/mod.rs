mod core;

pub use core::*;

use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterStep};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreAir =
    VmAirWrapper<Rv32LoadStoreAdapterAir, LoadStoreCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv32LoadStoreStep = LoadStoreStep<Rv32LoadStoreAdapterStep, RV32_REGISTER_NUM_LIMBS>;
pub type Rv32LoadStoreChip<F> = VmChipWrapper<F, LoadStoreFiller>;
