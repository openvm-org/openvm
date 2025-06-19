use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{Rv32BranchAdapterAir, Rv32BranchAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32BranchEqualChip =
    BranchEqualStep<Rv32BranchAdapterAir, Rv32BranchAdapterStep, RV32_REGISTER_NUM_LIMBS>;
