mod core;

pub use core::*;

use super::adapters::Rv32TerminateNopAdapterChip;
use crate::arch::VmChipWrapper;

#[cfg(test)]
// mod tests;

pub type Rv32TerminateNopChip<F> =
    VmChipWrapper<F, Rv32TerminateNopAdapterChip<F>, Rv32TerminateNopCoreChip<F>>;
