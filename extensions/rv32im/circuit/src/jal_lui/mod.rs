use crate::adapters::{Rv32CondRdWriteAdapterAir, Rv32CondRdWriteAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalLuiChip<F> =
    Rv32JalLuiStep<F, Rv32CondRdWriteAdapterAir, Rv32CondRdWriteAdapterStep>;
