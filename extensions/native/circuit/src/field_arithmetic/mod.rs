use crate::adapters::{AluNativeAdapterAir, AluNativeAdapterStep};
use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type FieldArithmeticAir = VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir>;
pub type FieldArithmeticStep = FieldArithmeticCoreStep<AluNativeAdapterStep>;
pub type FieldArithmeticChip<F> =
    NewVmChipWrapper<F, FieldArithmeticAir, FieldArithmeticStep, MatrixRecordArena<F>>;
