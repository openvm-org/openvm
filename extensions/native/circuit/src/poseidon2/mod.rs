use openvm_circuit::arch::VmChipWrapper;

pub mod air;
pub mod chip;
pub mod columns;
#[cfg(feature = "cuda")]
pub mod cuda;
mod execution;
#[cfg(test)]
mod tests;

use chip::NativePoseidon2Filler;

const CHUNK: usize = 8;
pub type NativePoseidon2Chip<F, const SBOX_REGISTERS: usize> =
    VmChipWrapper<F, NativePoseidon2Filler<F, SBOX_REGISTERS>>;
