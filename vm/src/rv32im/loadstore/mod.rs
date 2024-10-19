mod core;

pub use core::*;

use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32LoadStoreAdapterChip};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreChip<F> = VmChipWrapper<F, Rv32LoadStoreAdapterChip<F>, LoadStoreCoreChip<4>>;
