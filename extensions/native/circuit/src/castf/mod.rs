use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{ConvertAdapterAir, ConvertAdapterExecutor, ConvertAdapterFiller};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type CastFAir = VmAirWrapper<ConvertAdapterAir<1, 4>, CastFCoreAir>;
pub type CastFExecutor = CastFCoreExecutor<ConvertAdapterExecutor<1, 4>>;
pub type CastFChip<F> = VmChipWrapper<F, CastFCoreFiller<ConvertAdapterFiller<1, 4>>>;
