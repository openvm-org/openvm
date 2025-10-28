use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{
    NativeVectorizedAdapterAir, NativeVectorizedAdapterExecutor, NativeVectorizedAdapterFiller,
};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type FieldExtensionAir =
    VmAirWrapper<NativeVectorizedAdapterAir<EXT_DEG>, FieldExtensionCoreAir>;
pub type FieldExtensionExecutor =
    FieldExtensionCoreExecutor<NativeVectorizedAdapterExecutor<EXT_DEG>>;
pub type FieldExtensionChip<F> =
    VmChipWrapper<F, FieldExtensionCoreFiller<NativeVectorizedAdapterFiller<EXT_DEG>>>;
