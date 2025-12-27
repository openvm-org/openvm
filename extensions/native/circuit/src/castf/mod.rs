use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{ConvertAdapterAir, ConvertAdapterExecutor, ConvertAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type CastFAir = VmAirWrapper<ConvertAdapterAir<1, 4>, CastFCoreAir>;
pub type CastFExecutor = CastFCoreExecutor<ConvertAdapterExecutor<1, 4>>;
pub type CastFChip<F> = VmChipWrapper<F, CastFCoreFiller<ConvertAdapterFiller<1, 4>>>;
