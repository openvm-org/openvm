use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{Rv32BranchAdapterAir, Rv32BranchAdapterFiller, Rv32BranchAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualAir =
    VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv32BranchEqualStep = BranchEqualStep<Rv32BranchAdapterStep, RV32_REGISTER_NUM_LIMBS>;
pub type Rv32BranchEqualChip<F> =
    VmChipWrapper<F, BranchEqualFiller<Rv32BranchAdapterFiller, RV32_REGISTER_NUM_LIMBS>>;
