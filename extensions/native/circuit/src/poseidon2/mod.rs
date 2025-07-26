use openvm_circuit::arch::VmChipWrapper;

use crate::chip::NativePoseidon2Filler;

pub mod air;
pub mod chip;
pub mod columns;
#[cfg(test)]
mod tests;

const CHUNK: usize = 8;
pub type NativePoseidon2Chip<F, const SBOX_REGISTERS: usize> =
    VmChipWrapper<F, NativePoseidon2Filler<F, SBOX_REGISTERS>>;
