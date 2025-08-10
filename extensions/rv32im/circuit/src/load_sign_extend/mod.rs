use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterExecutor};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32LoadSignExtendAir = VmAirWrapper<
    Rv32LoadStoreAdapterAir,
    LoadSignExtendCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32LoadSignExtendExecutor =
    LoadSignExtendExecutor<Rv32LoadStoreAdapterExecutor, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32LoadSignExtendChip<F> = VmChipWrapper<F, LoadSignExtendFiller>;
