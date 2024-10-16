use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::adapters::native_adapter::{NativeAdapterAir, NativeAdapterChip},
};

#[cfg(test)]
pub mod tests;

mod core;
pub use core::*;

pub type ControlFlowAir = VmAirWrapper<NativeAdapterAir, ControlFlowCoreAir>;
pub type ControlFlowChip<F> = VmChipWrapper<F, NativeAdapterChip<F>, ControlFlowCoreChip>;
