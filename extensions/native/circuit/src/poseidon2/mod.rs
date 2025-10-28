use openvm_circuit::arch::VmChipWrapper;

pub mod air;
pub mod chip;
pub mod columns;
mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

use chip::NativePoseidon2Filler;

const CHUNK: usize = 8;
pub type NativePoseidon2Chip<F, const SBOX_REGISTERS: usize> =
    VmChipWrapper<F, NativePoseidon2Filler<F, SBOX_REGISTERS>>;
