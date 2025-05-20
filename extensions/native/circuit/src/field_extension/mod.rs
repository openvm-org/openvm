use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use super::adapters::native_vectorized_adapter::{
    NativeVectorizedAdapterAir, NativeVectorizedAdapterStep,
};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type FieldExtensionAir =
    VmAirWrapper<NativeVectorizedAdapterAir<EXT_DEG>, FieldExtensionCoreAir>;
pub type FieldExtensionStep = FieldExtensionCoreStep<NativeVectorizedAdapterStep<EXT_DEG>>;
pub type FieldExtensionChip<F> = NewVmChipWrapper<F, FieldExtensionAir, FieldExtensionStep>;
