use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{
    NativeVectorizedAdapterAir, NativeVectorizedAdapterFiller, NativeVectorizedAdapterStep,
};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type FieldExtensionAir =
    VmAirWrapper<NativeVectorizedAdapterAir<EXT_DEG>, FieldExtensionCoreAir>;
pub type FieldExtensionStep = FieldExtensionCoreStep<NativeVectorizedAdapterStep<EXT_DEG>>;
pub type FieldExtensionChip<F> =
    VmChipWrapper<F, FieldExtensionCoreFiller<NativeVectorizedAdapterFiller<EXT_DEG>>>;
