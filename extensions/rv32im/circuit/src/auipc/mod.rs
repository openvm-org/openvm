use crate::adapters::{Rv32RdWriteAdapterAir, Rv32RdWriteAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32AuipcChip<F> = Rv32AuipcStep<F, Rv32RdWriteAdapterAir, Rv32RdWriteAdapterStep>;
