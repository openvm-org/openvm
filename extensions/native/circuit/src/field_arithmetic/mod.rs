use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{AluNativeAdapterAir, AluNativeAdapterExecutor, AluNativeAdapterFiller};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type FieldArithmeticAir = VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir>;
pub type FieldArithmeticExecutor = FieldArithmeticCoreExecutor<AluNativeAdapterExecutor>;
pub type FieldArithmeticChip<F> =
    VmChipWrapper<F, FieldArithmeticCoreFiller<AluNativeAdapterFiller>>;
