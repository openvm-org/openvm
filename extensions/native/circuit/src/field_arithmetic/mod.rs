use openvm_circuit::{
    arch::{VmAirWrapper, VmChipWrapper},
    system::native_adapter::{NativeAdapterAir, NativeAdapterChip},
};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type FieldArithmeticAir = VmAirWrapper<NativeAdapterAir<2, 1>, FieldArithmeticCoreAir>;
pub type FieldArithmeticChip<F> =
    VmChipWrapper<F, NativeAdapterChip<F, 2, 1>, FieldArithmeticCoreChip>;
