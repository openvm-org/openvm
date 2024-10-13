mod core;

pub use core::*;

use crate::arch::{Rv32LoadStoreAdapter, VmChipWrapper};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreChip<F> = VmChipWrapper<F, Rv32LoadStoreAdapter<F, 4>, LoadStoreCoreChip<F, 4>>;
