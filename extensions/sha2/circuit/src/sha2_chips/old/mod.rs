//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_sha256_air::{Sha256FillerHelper, SHA256_BLOCK_BITS};
use sha2::{Digest, Sha256};

mod air;
mod columns;
mod execution;
mod trace;

pub use air::*;
pub use columns::*;
pub use trace::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

// ==== Constants for register/memory adapter ====
/// Register reads to get dst, src, len
const SHA256_REGISTER_READS: usize = 3;
/// Number of cells to read in a single memory access
const SHA256_READ_SIZE: usize = 16;
/// Number of cells to write in a single memory access
const SHA256_WRITE_SIZE: usize = 32;
/// Number of rv32 cells read in a SHA256 block
pub const SHA256_BLOCK_CELLS: usize = SHA256_BLOCK_BITS / RV32_CELL_BITS;
/// Number of rows we will do a read on for each SHA256 block
pub const SHA256_NUM_READ_ROWS: usize = SHA256_BLOCK_CELLS / SHA256_READ_SIZE;
/// Maximum message length that this chip supports in bytes
pub const SHA256_MAX_MESSAGE_LEN: usize = 1 << 29;

pub type Sha256VmChip<F> = VmChipWrapper<F, Sha256VmFiller>;

#[derive(derive_new::new, Clone)]
pub struct Sha256VmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

pub struct Sha256VmFiller {
    pub inner: Sha256FillerHelper,
    pub padding_encoder: Encoder,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub pointer_max_bits: usize,
}

impl Sha256VmFiller {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        pointer_max_bits: usize,
    ) -> Self {
        Self {
            inner: Sha256FillerHelper::new(),
            padding_encoder: Encoder::new(PaddingFlags::COUNT, 2, false),
            bitwise_lookup_chip,
            pointer_max_bits,
        }
    }
}

pub fn sha256_solve(input_message: &[u8]) -> [u8; SHA256_WRITE_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(input_message);
    let mut output = [0u8; SHA256_WRITE_SIZE];
    output.copy_from_slice(hasher.finalize().as_ref());
    output
}
