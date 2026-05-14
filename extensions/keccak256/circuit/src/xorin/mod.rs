pub mod air;
pub mod columns;
pub mod execution;
#[cfg(test)]
pub mod tests;
pub mod trace;

use openvm_circuit::arch::VmChipWrapper;
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, var_range::SharedVariableRangeCheckerChip,
};

#[derive(derive_new::new, Clone, Copy)]
pub struct XorinVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[derive(derive_new::new)]
// number of bits = 8
pub struct XorinVmFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    /// Used to range-check the u16 high cells of `buffer_ptr` and `input_ptr`.
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub pointer_max_bits: usize,
}

pub type XorinVmChip<F> = VmChipWrapper<F, XorinVmFiller>;
