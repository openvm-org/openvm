use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::convert_adapter::{ConvertAdapterAir, ConvertAdapterChip};

#[cfg(test)]
mod tests;

mod core;
pub use core::*;

pub type CastFAir = VmAirWrapper<ConvertAdapterAir<1, 4>, CastFCoreAir>;
pub type CastFChip<F> = VmChipWrapper<F, ConvertAdapterChip<F, 1, 4>, CastFCoreChip>;
