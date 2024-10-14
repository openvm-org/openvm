mod core;

pub use core::*;

use crate::{arch::VmChipWrapper, rv32im::adapters::Rv32LoadStoreAdapter};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreChip<F> =
    VmChipWrapper<F, Rv32LoadStoreAdapter<F, 4>, LoadStoreCoreChip<F, 4>>;
