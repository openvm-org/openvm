#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
//! BLAKE3 hasher circuit extension for OpenVM.
//! Handles BLAKE3 hashing on variable length inputs read from VM memory.

use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;

pub mod air;
pub mod columns;
pub mod execution;
pub mod trace;
pub mod utils;

mod extension;
#[cfg(test)]
mod tests;
pub use air::Blake3VmAir;
pub use extension::*;
use openvm_circuit::arch::*;

// ==== Constants for register/memory adapter ====
/// Register reads to get dst, src, len
const BLAKE3_REGISTER_READS: usize = 3;
/// Number of cells to read/write in a single memory access
pub const BLAKE3_WORD_SIZE: usize = 4;

// ==== BLAKE3 constants ====
/// BLAKE3 block size in bytes
pub const BLAKE3_BLOCK_BYTES: usize = 64;
/// BLAKE3 output size in bytes
pub const BLAKE3_DIGEST_BYTES: usize = 32;
/// BLAKE3 chaining value size in bytes
pub const BLAKE3_CV_BYTES: usize = 32;
/// Number of u32 words in BLAKE3 state
pub const BLAKE3_STATE_WORDS: usize = 16;
/// Number of u32 words in BLAKE3 message block
pub const BLAKE3_BLOCK_WORDS: usize = 16;
/// Number of rounds in BLAKE3 compression
pub const BLAKE3_ROUNDS: usize = 7;

/// Memory reads for input per row
const BLAKE3_INPUT_READS: usize = BLAKE3_BLOCK_BYTES / BLAKE3_WORD_SIZE;
/// Memory writes for digest
const BLAKE3_DIGEST_WRITES: usize = BLAKE3_DIGEST_BYTES / BLAKE3_WORD_SIZE;

pub type Blake3VmChip<F> = VmChipWrapper<F, Blake3VmFiller>;

#[derive(derive_new::new, Clone, Copy)]
pub struct Blake3VmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Blake3VmFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub pointer_max_bits: usize,
}
