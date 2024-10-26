mod core;

pub use core::*;

use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32HintStoreAdapterChip};

#[cfg(test)]
mod tests;

pub type Rv32HintStoreChip<F> =
    VmChipWrapper<F, Rv32HintStoreAdapterChip<F>, Rv32HintStoreCoreChip<F>>;
