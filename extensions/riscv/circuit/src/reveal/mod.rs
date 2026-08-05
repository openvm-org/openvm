use openvm_circuit::arch::VmChipWrapper;
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;

mod air;
mod execution;
pub(crate) mod trace;

pub use air::*;
pub use execution::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type RevealChip<F> = VmChipWrapper<F, RevealFiller>;

/// Trace filler for the standalone public-value reveal instruction.
#[derive(Clone, derive_new::new)]
pub struct RevealFiller {
    pub(crate) range_checker_chip: SharedVariableRangeCheckerChip,
    pub(crate) timestamp_max_bits: usize,
}
