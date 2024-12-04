use axvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::jal_native_adapter::{JalNativeAdapterAir, JalNativeAdapterChip};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type KernelJalAir = VmAirWrapper<JalNativeAdapterAir, JalCoreAir>;
pub type KernelJalChip<F> = VmChipWrapper<F, JalNativeAdapterChip<F>, JalCoreChip>;
