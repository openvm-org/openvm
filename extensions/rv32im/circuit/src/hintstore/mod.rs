use openvm_circuit::arch::VmChipWrapper;

use crate::adapters::Rv32HintStoreAdapterChip;

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32HintStoreChip<F> =
    VmChipWrapper<F, Rv32HintStoreAdapterChip<F>, Rv32HintStoreCoreChip<F>>;
