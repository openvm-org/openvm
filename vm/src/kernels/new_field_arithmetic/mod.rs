use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::{
        adapters::native_adapter::{NativeAdapterAir, NativeAdapterChip},
        new_field_arithmetic::{NewFieldArithmeticCoreAir, NewFieldArithmeticCoreChip},
    },
};

#[cfg(test)]
pub mod tests;

mod core;
pub use core::*;

pub type NewFieldArithmeticAir = VmAirWrapper<NativeAdapterAir, NewFieldArithmeticCoreAir>;
pub type NewFieldArithmeticChip<F> =
    VmChipWrapper<F, NativeAdapterChip<F>, NewFieldArithmeticCoreChip>;
