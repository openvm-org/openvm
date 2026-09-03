use openvm_circuit::arch::VmChipWrapper;
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, var_range::SharedVariableRangeCheckerChip,
};
use openvm_instructions::riscv::BYTE_BITS;

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

#[derive(Clone, derive_new::new)]
pub struct RevealFiller {
    pub(crate) pointer_max_bits: usize,
    pub(crate) range_checker_chip: SharedVariableRangeCheckerChip,
    pub(crate) bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
}
