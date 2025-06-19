use crate::adapters::{Rv32JalrAdapterAir, Rv32JalrAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type Rv32JalrChip = Rv32JalrStep<Rv32JalrAdapterAir, Rv32JalrAdapterStep>;
