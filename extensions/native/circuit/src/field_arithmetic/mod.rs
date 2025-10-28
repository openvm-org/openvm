use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{AluNativeAdapterAir, AluNativeAdapterExecutor, AluNativeAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type FieldArithmeticAir = VmAirWrapper<AluNativeAdapterAir, FieldArithmeticCoreAir>;
pub type FieldArithmeticExecutor = FieldArithmeticCoreExecutor<AluNativeAdapterExecutor>;
pub type FieldArithmeticChip<F> =
    VmChipWrapper<F, FieldArithmeticCoreFiller<AluNativeAdapterFiller>>;
