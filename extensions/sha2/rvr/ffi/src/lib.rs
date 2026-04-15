//! FFI functions for the SHA-2 extension (SHA-256 + SHA-512).
//!
//! These Rust functions are called from generated C code. They receive resolved
//! register values and the state as an opaque pointer. Memory reads/writes,
//! SHA-2 computation, and chip tracing go through double FFI; register access
//! stays on the C side.

use std::ffi::c_void;

use generic_array::GenericArray;
use rvr_openvm_ext_ffi_common::{
    rd_mem_words_traced, trace_chip_wrapper, u32s_as_u8s, u64s_as_u32s, u64s_as_u32s_mut,
    wr_mem_words_traced, WORD_SIZE,
};
use sha2::compress256;
use sha2::compress512;

// SHA-256 constants
const SHA256_STATE_BYTES: usize = 32;
const SHA256_BLOCK_BYTES: usize = 64;
const SHA256_STATE_WORDS: usize = SHA256_STATE_BYTES / WORD_SIZE;
const SHA256_BLOCK_WORDS: usize = SHA256_BLOCK_BYTES / WORD_SIZE;
const SHA256_ROWS_PER_BLOCK: u32 = 17;

// SHA-512 constants
const SHA512_BLOCK_BYTES: usize = 128;
const SHA512_BLOCK_WORDS: usize = SHA512_BLOCK_BYTES / WORD_SIZE;
const SHA512_ROWS_PER_BLOCK: u32 = 21;

/// Both SHA-256 and SHA-512 use 8-element internal state arrays.
const HASH_STATE_LEN: usize = 8;

/// SHA-256 compress FFI entry point.
///
/// Reads `SHA256_STATE_BYTES` of state and `SHA256_BLOCK_BYTES` of input block,
/// applies SHA-256 compression, writes new state to `dst_ptr`. Reports chip
/// heights for metering.
///
/// # Safety
///
/// `state` must be a valid pointer to the C `RvState` struct.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_sha256(
    state: *mut c_void,
    dst_ptr: u32,
    state_ptr: u32,
    input_ptr: u32,
    _main_chip_idx: u32,
    block_hasher_chip_idx: u32,
) {
    // Read state (8 u32) and input block (16 u32) from memory.
    let mut state_words = [0u32; SHA256_STATE_WORDS];
    rd_mem_words_traced(state, state_ptr, &mut state_words);
    let mut block_words = [0u32; SHA256_BLOCK_WORDS];
    rd_mem_words_traced(state, input_ptr, &mut block_words);

    // compress256 wants the block as `GenericArray<u8, U64>`. Reinterpret the
    // u32 buffer as bytes (zero-copy on LE).
    let block_array: &GenericArray<u8, _> = GenericArray::from_slice(u32s_as_u8s(&block_words));
    compress256(&mut state_words, &[*block_array]);

    wr_mem_words_traced(state, dst_ptr, &state_words);

    // Trace chip height for metering.
    // The per-instruction Sha2Main cost (1 row) is covered by the per-block
    // chip update emitted at block entry.
    // Only the additional Sha2BlockHasher rows need trace_chip.
    trace_chip_wrapper(state, block_hasher_chip_idx, SHA256_ROWS_PER_BLOCK);
}

/// SHA-512 compress FFI entry point.
///
/// Reads `SHA512_STATE_BYTES` of state and `SHA512_BLOCK_BYTES` of input block,
/// applies SHA-512 compression, writes new state to `dst_ptr`. Reports chip
/// heights for metering.
///
/// # Safety
///
/// `state` must be a valid pointer to the C `RvState` struct.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_sha512(
    state: *mut c_void,
    dst_ptr: u32,
    state_ptr: u32,
    input_ptr: u32,
    _main_chip_idx: u32,
    block_hasher_chip_idx: u32,
) {
    // u64-aligned scratch holds the SHA-512 state; reinterpret as u32 words at
    // the FFI boundary (zero-copy on LE).
    let mut state_u64s = [0u64; HASH_STATE_LEN];
    rd_mem_words_traced(state, state_ptr, u64s_as_u32s_mut(&mut state_u64s));
    let mut block_words = [0u32; SHA512_BLOCK_WORDS];
    rd_mem_words_traced(state, input_ptr, &mut block_words);

    let block_array: &GenericArray<u8, _> = GenericArray::from_slice(u32s_as_u8s(&block_words));
    compress512(&mut state_u64s, &[*block_array]);

    wr_mem_words_traced(state, dst_ptr, u64s_as_u32s(&state_u64s));

    // Trace chip height for metering.
    trace_chip_wrapper(state, block_hasher_chip_idx, SHA512_ROWS_PER_BLOCK);
}
