mod core;

pub use core::*;

use super::adapters::RV32_REGISTER_NUM_LANES;
use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32LoadStoreAdapterChip};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreChip<F> =
    VmChipWrapper<F, Rv32LoadStoreAdapterChip<F>, LoadStoreCoreChip<RV32_REGISTER_NUM_LANES>>;
