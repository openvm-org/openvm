mod air;
mod columns;
pub mod execution;
#[cfg(test)]
pub mod tests;
mod trace;

use openvm_circuit::arch::VmChipWrapper;
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;

#[derive(derive_new::new, Clone, Copy)]
pub struct KeccakfVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct KeccakfVmFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub pointer_max_bits: usize,
}

pub type KeccakfVmChip<F> = VmChipWrapper<F, KeccakfVmFiller>;
