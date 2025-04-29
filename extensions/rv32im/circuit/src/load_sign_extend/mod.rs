use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32LoadSignExtendAir = VmAirWrapper<
    Rv32LoadStoreAdapterAir,
    LoadSignExtendCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32LoadSignExtendStep =
    LoadSignExtendStep<Rv32LoadStoreAdapterStep, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32LoadSignExtendChip<F> =
    NewVmChipWrapper<F, Rv32LoadSignExtendAir, Rv32LoadSignExtendStep>;
