use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::convert_adapter::{ConvertAdapterAir, ConvertAdapterChip};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type CastFAir = VmAirWrapper<ConvertAdapterAir<1, 4>, CastFCoreAir>;
pub type CastFStepWithAdapter = CastFStep<ConvertAdapterStep<1, 4>>;
pub type CastFChip<F> = NewVmChipWrapper<F, CastFAir, CastFStepWithAdapter>;
