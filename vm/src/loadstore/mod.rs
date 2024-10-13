mod integration;

pub use integration::*;

use crate::arch::{VmChipWrapper, Rv32LoadStoreAdapter};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreChip<F> =
    VmChipWrapper<F, Rv32LoadStoreAdapter<F, 4>, LoadStoreIntegration<F, 4>>;
