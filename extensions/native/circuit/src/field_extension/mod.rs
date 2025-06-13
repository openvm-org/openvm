use crate::adapters::{NativeVectorizedAdapterAir, NativeVectorizedAdapterStep};
use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type FieldExtensionAir =
    VmAirWrapper<NativeVectorizedAdapterAir<EXT_DEG>, FieldExtensionCoreAir>;
pub type FieldExtensionStep = FieldExtensionCoreStep<NativeVectorizedAdapterStep<EXT_DEG>>;
pub type FieldExtensionChip<F> =
    NewVmChipWrapper<F, FieldExtensionAir, FieldExtensionStep, MatrixRecordArena<F>>;
