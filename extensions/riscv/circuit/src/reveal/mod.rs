use openvm_circuit::arch::VmChipWrapper;
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;

mod air;
mod execution;
pub(crate) mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub use air::*;
pub use execution::*;

pub type RevealChip<F> = VmChipWrapper<F, RevealFiller>;

#[derive(Clone)]
pub struct RevealFiller {
    pub(crate) pointer_max_bits: usize,
    pub(crate) range_checker: SharedVariableRangeCheckerChip,
}

impl RevealFiller {
    pub fn new(pointer_max_bits: usize, range_checker: SharedVariableRangeCheckerChip) -> Self {
        Self {
            pointer_max_bits,
            range_checker,
        }
    }
}
