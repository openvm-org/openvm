use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use super::adapters::convert_adapter::{ConvertAdapterAir, ConvertAdapterStep};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type CastFAir = VmAirWrapper<ConvertAdapterAir<1, 4>, CastFCoreAir>;
pub type CastFStep = CastFCoreStep<ConvertAdapterStep<1, 4>>;
pub type CastFChip<F> = NewVmChipWrapper<F, CastFAir, CastFStep>;
