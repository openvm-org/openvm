//! FFI functions for the SHA-2 extension (SHA-256 + SHA-512).
//!
//! These Rust functions are called from generated C code. They receive resolved
//! register values and the state as an opaque pointer. Memory reads/writes,
//! and SHA-2 computation go through double FFI; register access stays on the C
//! side. Fixed chip-row accounting is folded into generated block metering.

use std::ffi::c_void;

use generic_array::GenericArray;
use openvm_platform::WORD_SIZE;
use rvr_openvm_ext_ffi_common::{rd_mem_words_traced, u64s_as_u32s_mut, wr_mem_words_traced};
use sha2::{compress256, compress512};

// SHA-256: 32-byte state (4 u64s / 8 u32s), 64-byte block (8 u64s)
const SHA256_STATE_BYTES: usize = 32;
const SHA256_BLOCK_BYTES: usize = 64;
const SHA256_STATE_WORDS: usize = SHA256_STATE_BYTES / WORD_SIZE;
const SHA256_BLOCK_WORDS: usize = SHA256_BLOCK_BYTES / WORD_SIZE;

// SHA-512: 64-byte state (8 u64s), 128-byte block (16 u64s)
const SHA512_STATE_BYTES: usize = 64;
const SHA512_BLOCK_BYTES: usize = 128;
const SHA512_STATE_WORDS: usize = SHA512_STATE_BYTES / WORD_SIZE;
const SHA512_BLOCK_WORDS: usize = SHA512_BLOCK_BYTES / WORD_SIZE;

#[inline(always)]
fn u64_words_as_bytes(words: &[u64]) -> &[u8] {
    let len = std::mem::size_of_val(words);
    // SAFETY: u64 alignment is stricter than u8 alignment, total bytes match,
    // and the FFI backend is only supported on little-endian hosts.
    unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), len) }
}

/// SHA-256 compress FFI entry point.
///
/// Reads `SHA256_STATE_BYTES` of state and `SHA256_BLOCK_BYTES` of input block,
/// applies SHA-256 compression, and writes new state to `dst_ptr`.
///
/// # Safety
///
/// `state` must be a valid pointer to the C `RvState` struct.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_sha256(
    state: *mut c_void,
    dst_ptr: u64,
    state_ptr: u64,
    input_ptr: u64,
) {
    // Read state as 4 u64s (32 bytes); view as [u32; 8] for compress256.
    let mut state_words = [0u64; SHA256_STATE_WORDS];
    rd_mem_words_traced(state, state_ptr, &mut state_words);
    let state_u32s: &mut [u32; 8] = u64s_as_u32s_mut(&mut state_words).try_into().unwrap();

    // Read block as 8 u64s (64 bytes); view as bytes for compress256.
    let mut block_words = [0u64; SHA256_BLOCK_WORDS];
    rd_mem_words_traced(state, input_ptr, &mut block_words);
    let block_array: &GenericArray<u8, _> =
        GenericArray::from_slice(u64_words_as_bytes(&block_words));

    compress256(state_u32s, &[*block_array]);
    // state_u32s borrow ends here; state_words now holds the compressed state.

    wr_mem_words_traced(state, dst_ptr, &state_words);
}

/// SHA-512 compress FFI entry point.
///
/// Reads `SHA512_STATE_BYTES` of state and `SHA512_BLOCK_BYTES` of input block,
/// applies SHA-512 compression, and writes new state to `dst_ptr`.
///
/// # Safety
///
/// `state` must be a valid pointer to the C `RvState` struct.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_sha512(
    state: *mut c_void,
    dst_ptr: u64,
    state_ptr: u64,
    input_ptr: u64,
) {
    // SHA-512 state is naturally [u64; 8]; read directly.
    let mut state_u64s = [0u64; SHA512_STATE_WORDS];
    rd_mem_words_traced(state, state_ptr, &mut state_u64s);

    // Read block as 16 u64s (128 bytes); view as bytes for compress512.
    let mut block_words = [0u64; SHA512_BLOCK_WORDS];
    rd_mem_words_traced(state, input_ptr, &mut block_words);
    let block_array: &GenericArray<u8, _> =
        GenericArray::from_slice(u64_words_as_bytes(&block_words));

    compress512(&mut state_u64s, &[*block_array]);

    wr_mem_words_traced(state, dst_ptr, &state_u64s);
}
