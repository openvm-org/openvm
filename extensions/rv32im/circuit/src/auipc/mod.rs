use crate::adapters::{Rv32RdWriteAdapterAir, Rv32RdWriteAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32AuipcChip = Rv32AuipcStep<Rv32RdWriteAdapterAir, Rv32RdWriteAdapterStep>;
