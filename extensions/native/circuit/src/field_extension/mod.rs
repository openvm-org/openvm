use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{NativeVectorizedAdapterAir, NativeVectorizedAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type FieldExtensionAir =
    VmAirWrapper<NativeVectorizedAdapterAir<EXT_DEG>, FieldExtensionCoreAir>;
pub type FieldExtensionStep = FieldExtensionCoreStep<NativeVectorizedAdapterStep<EXT_DEG>>;
pub type FieldExtensionChip<F> =
    NewVmChipWrapper<F, FieldExtensionAir, FieldExtensionStep, MatrixRecordArena<F>>;
