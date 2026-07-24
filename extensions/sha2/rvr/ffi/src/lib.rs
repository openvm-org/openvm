//! FFI functions for the SHA-2 extension (SHA-256 + SHA-512).
//!
//! Generated C calls these Rust functions, which call back into C for memory
//! access. Register access stays in generated C. The generator adds fixed chip
//! rows to each block's metering update.

use std::ffi::c_void;

use generic_array::GenericArray;
use openvm_platform::WORD_SIZE;
use rvr_openvm_ext_ffi_common::{read_mem_words, u64s_as_u32s_mut, write_mem_words};
use sha2::{compress256, compress512};

extern "C" {
    fn rvr_ext_sha2_is_preflight() -> u8;
}

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

/// SHA-256 fallback FFI entry point.
///
/// Reads `SHA256_STATE_BYTES` of state and `SHA256_BLOCK_BYTES` of input block,
/// applies SHA-256 compression, and writes new state to `dst_ptr`.
///
/// # Safety
///
/// `state` must be a valid pointer to the C `RvState` struct.
#[no_mangle]
pub unsafe extern "C" fn rvr_ext_sha256_fallback(
    state: *mut c_void,
    dst_ptr: u64,
    state_ptr: u64,
    input_ptr: u64,
) {
    let mut state_words = [0u64; SHA256_STATE_WORDS];
    let mut block_words = [0u64; SHA256_BLOCK_WORDS];
    if rvr_ext_sha2_is_preflight() != 0 {
        // The circuit timestamps input reads before state reads.
        read_mem_words(state, input_ptr, &mut block_words);
        read_mem_words(state, state_ptr, &mut state_words);
    } else {
        // Preserve the established pure/metered access order.
        read_mem_words(state, state_ptr, &mut state_words);
        read_mem_words(state, input_ptr, &mut block_words);
    }
    let state_u32s: &mut [u32; 8] = u64s_as_u32s_mut(&mut state_words).try_into().unwrap();
    let block_array: &GenericArray<u8, _> =
        GenericArray::from_slice(u64_words_as_bytes(&block_words));

    compress256(state_u32s, &[*block_array]);
    // state_u32s borrow ends here; state_words now holds the compressed state.

    write_mem_words(state, dst_ptr, &state_words);
}

/// Applies SHA-256 compression to native little-endian memory words.
///
/// This computation-only entry point lets generated preflight C own the exact
/// memory chronology and direct-final record writes without duplicating the
/// SHA-256 compression implementation.
///
/// # Safety
///
/// `state_words` and `block_words` must point to arrays of 4 and 8 `u64`s,
/// respectively.
#[no_mangle]
pub unsafe extern "C" fn rvr_sha256_compress_words(state_words: *mut u64, block_words: *const u64) {
    let state_words = unsafe { &mut *(state_words as *mut [u64; SHA256_STATE_WORDS]) };
    let block_words = unsafe { &*(block_words as *const [u64; SHA256_BLOCK_WORDS]) };
    let state_u32s: &mut [u32; 8] = u64s_as_u32s_mut(state_words).try_into().unwrap();
    let block_array: &GenericArray<u8, _> =
        GenericArray::from_slice(u64_words_as_bytes(block_words));
    compress256(state_u32s, &[*block_array]);
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
    read_mem_words(state, state_ptr, &mut state_u64s);

    // Read block as 16 u64s (128 bytes); view as bytes for compress512.
    let mut block_words = [0u64; SHA512_BLOCK_WORDS];
    read_mem_words(state, input_ptr, &mut block_words);
    let block_array: &GenericArray<u8, _> =
        GenericArray::from_slice(u64_words_as_bytes(&block_words));

    compress512(&mut state_u64s, &[*block_array]);

    write_mem_words(state, dst_ptr, &state_u64s);
}
