mod core;

pub use core::*;

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LANES};
use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32LoadStoreAdapterChip};

#[cfg(test)]
mod tests;

pub type Rv32LoadSignExtendChip<F> = VmChipWrapper<
    F,
    Rv32LoadStoreAdapterChip<F>,
    LoadSignExtendCoreChip<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>,
>;
