pub mod air;
pub mod columns;
pub mod execution;
#[cfg(all(test, feature = "test-utils"))]
pub mod tests;
pub mod trace;

#[cfg(feature = "cuda")]
pub mod cuda {}

use openvm_circuit::arch::VmChipWrapper;
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;

const KECCAK_WORD_SIZE: usize = 4;

#[derive(derive_new::new, Clone, Copy)]
pub struct XorinVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[derive(derive_new::new)]
// number of bits = 8
pub struct XorinVmFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub pointer_max_bits: usize,
}

pub type XorinVmChip<F> = VmChipWrapper<F, XorinVmFiller>;
