use openvm_circuit::arch::{VmAirWrapper};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{Rv32BranchAdapterAir, Rv32BranchAdapterStep, Rv32BranchAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualAir =
    VmAirWrapper<Rv32BranchAdapterAir, BranchEqualCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv32BranchEqualStep = BranchEqualStep<Rv32BranchAdapterStep, RV32_REGISTER_NUM_LIMBS>;
pub type Rv32BranchEqualChip<F> = BranchEqualChip<F, Rv32BranchAdapterChip, RV32_REGISTER_NUM_LIMBS>;
