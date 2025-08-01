use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{AluNativeAdapterAir, AluNativeAdapterFiller, AluNativeAdapterStep};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type FieldArithmeticAir = VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir>;
pub type FieldArithmeticStep = FieldArithmeticCoreStep<AluNativeAdapterStep>;
pub type FieldArithmeticChip<F> =
    VmChipWrapper<F, FieldArithmeticCoreFiller<AluNativeAdapterFiller>>;
