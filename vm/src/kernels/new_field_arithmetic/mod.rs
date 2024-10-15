use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::adapters::native_adapter::{NativeAdapterAir, NativeAdapterChip},
};

#[cfg(test)]
pub mod tests;

mod core;
pub use core::*;

pub type NewFieldArithmeticAir = VmAirWrapper<NativeAdapterAir, NewFieldArithmeticCoreAir>;
pub type NewFieldArithmeticChip<F> =
    VmChipWrapper<F, NativeAdapterChip<F>, NewFieldArithmeticCoreChip>;
