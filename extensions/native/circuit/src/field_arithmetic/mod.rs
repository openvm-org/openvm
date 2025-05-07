use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper, VmChipWrapper};

use crate::adapters::alu_native_adapter::{AluNativeAdapterAir, AluNativeAdapterStep};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type FieldArithmeticAir = VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir>;
pub type FieldArithmeticStepWithAdapter = FieldArithmeticStep<AluNativeAdapterStep>;
pub type FieldArithmeticChip<F> =
    NewVmChipWrapper<F, FieldArithmeticAir, FieldArithmeticStepWithAdapter>;
