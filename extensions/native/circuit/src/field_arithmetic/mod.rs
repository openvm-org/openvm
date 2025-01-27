use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::alu_native_adapter::{AluNativeAdapterAir, AluNativeAdapterChip};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type FieldArithmeticAir = VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir>;
pub type FieldArithmeticChip<F> =
    VmChipWrapper<F, AluNativeAdapterChip<F>, FieldArithmeticCoreChip>;
