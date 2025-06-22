use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};

use crate::adapters::{ConvertAdapterAir, ConvertAdapterStep};

mod core;
pub use core::*;

#[cfg(test)]
mod tests;

pub type CastFAir = VmAirWrapper<ConvertAdapterAir<1, 4>, CastFCoreAir>;
pub type CastFStep = CastFCoreStep<ConvertAdapterStep<1, 4>>;
pub type CastFChip<F> = NewVmChipWrapper<F, CastFAir, CastFStep, MatrixRecordArena<F>>;
