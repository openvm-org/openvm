use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::jal_native_adapter::{JalNativeAdapterAir, JalNativeAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type NativeJalAir = VmAirWrapper<JalNativeAdapterAir, JalCoreAir>;
pub type NativeJalChip<F> = VmChipWrapper<F, JalNativeAdapterChip<F>, JalCoreChip>;
