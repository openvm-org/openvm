use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use super::adapters::native_vectorized_adapter::{
    NativeVectorizedAdapterAir, NativeVectorizedAdapterChip,
};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type FieldExtensionAir =
    VmAirWrapper<NativeVectorizedAdapterAir<EXT_DEG>, FieldExtensionCoreAir>;
pub type FieldExtensionStep = FieldExtensionCoreStep<NativeVectorizedAdapterChip<EXT_DEG>>;
pub type FieldExtensionChip<F> = NewVmChipWrapper<F, FieldExtensionAir, FieldExtensionStep>;
