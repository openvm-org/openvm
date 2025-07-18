use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{ConvertAdapterAir, ConvertAdapterFiller, ConvertAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type CastFAir = VmAirWrapper<ConvertAdapterAir<1, 4>, CastFCoreAir>;
pub type CastFStep = CastFCoreStep<ConvertAdapterStep<1, 4>>;
pub type CastFChip<F> = VmChipWrapper<F, CastFCoreFiller<ConvertAdapterFiller<1, 4>>>;
