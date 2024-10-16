use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    kernels::adapters::native_vectorized_adapter::{
        NativeVectorizedAdapterAir, NativeVectorizedAdapterChip,
    },
};

#[cfg(test)]
pub mod tests;

mod core;
pub use core::*;

pub type FieldExtensionAir =
    VmAirWrapper<NativeVectorizedAdapterAir<EXT_DEG>, FieldExtensionCoreAir>;
pub type FieldExtensionChip<F> =
    VmChipWrapper<F, NativeVectorizedAdapterChip<F, EXT_DEG>, FieldExtensionCoreChip>;
