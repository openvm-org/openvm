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

pub type NewFieldExtensionAir =
    VmAirWrapper<NativeVectorizedAdapterAir<EXT_DEG>, NewFieldExtensionCoreAir>;
pub type NewFieldExtensionChip<F> =
    VmChipWrapper<F, NativeVectorizedAdapterChip<F, EXT_DEG>, NewFieldExtensionCoreChip>;
