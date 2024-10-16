use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::adapters::native_adapter::{NativeAdapterAir, NativeAdapterChip},
};

#[cfg(test)]
pub mod tests;

mod core;
pub use core::*;

pub type FieldArithmeticAir = VmAirWrapper<NativeAdapterAir, FieldArithmeticCoreAir>;
pub type FieldArithmeticChip<F> = VmChipWrapper<F, NativeAdapterChip<F>, FieldArithmeticCoreChip>;
