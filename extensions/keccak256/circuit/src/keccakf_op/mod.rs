mod air;
pub mod columns;
mod execution;
#[cfg(test)]
pub mod tests;
/// Preflight and CPU trace generation
pub mod trace;

use std::mem::MaybeUninit;

pub use air::*;
pub use columns::*;
pub use trace::*;

use crate::{KECCAK_WIDTH_BYTES, KECCAK_WIDTH_U64S};

pub const NUM_OP_ROWS_PER_INS: usize = 2;

#[derive(derive_new::new, Clone, Copy)]
pub struct KeccakfExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[cfg(target_endian = "little")]
fn copy_bytes_to_u64(src: &[u8; KECCAK_WIDTH_BYTES]) -> [u64; KECCAK_WIDTH_U64S] {
    // 1. Create uninitialized memory for the destination. This avoids the performance cost of
    //    writing zeros first.
    let mut dst = MaybeUninit::<[u64; KECCAK_WIDTH_U64S]>::uninit();

    unsafe {
        // 2. Perform a raw memory copy. We cast the destination pointer to *mut u8 to copy
        //    byte-by-byte. This is crucial: copying as u8s ignores the alignment of 'src'.
        std::ptr::copy_nonoverlapping(
            src.as_ptr(),                // Source pointer (*const u8)
            dst.as_mut_ptr() as *mut u8, // Destination pointer (cast to *mut u8)
            KECCAK_WIDTH_BYTES,          // Number of BYTES to copy
        );
        // 3. Assume initialization is complete. We know we filled all 200 bytes (25 * 8), so this
        //    is safe.
        dst.assume_init()
    }
}

#[cfg(not(target_endian = "little"))]
fn copy_bytes_to_u64(src: &[u8; KECCAK_WIDTH_BYTES]) -> [u64; KECCAK_WIDTH_U64S] {
    let mut dst = [0u64; KECCAK_WIDTH_U64S];
    for (u64_word, chunk) in dst.iter_mut().zip(src.chunks_exact(8)) {
        *u64_word = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    dst
}

#[cfg(target_endian = "little")]
fn transmute_u64_to_bytes(src: [u64; KECCAK_WIDTH_U64S]) -> [u8; KECCAK_WIDTH_BYTES] {
    // SAFETY:
    // - The size of both is the same, and `src` is plain old data (so we do not even need the fact
    //   that src has alignment > alignment of dst).
    unsafe { std::mem::transmute::<[u64; KECCAK_WIDTH_U64S], [u8; KECCAK_WIDTH_BYTES]>(src) }
}

#[cfg(not(target_endian = "little"))]
fn transmute_u64_to_bytes(src: [u64; KECCAK_WIDTH_U64S]) -> [u8; KECCAK_WIDTH_BYTES] {
    let mut dst = [0u8; KECCAK_WIDTH_BYTES];
    for (chunk, u64_word) in dst.chunks_exact_mut(8).zip(src.iter()) {
        chunk.copy_from_slice(&u64_word.to_le_bytes());
    }
    dst
}

fn keccakf_postimage_bytes(
    preimage_buffer_bytes: &[u8; KECCAK_WIDTH_BYTES],
) -> [u8; KECCAK_WIDTH_BYTES] {
    let mut state = copy_bytes_to_u64(preimage_buffer_bytes);
    tiny_keccak::keccakf(&mut state);
    transmute_u64_to_bytes(state)
}
