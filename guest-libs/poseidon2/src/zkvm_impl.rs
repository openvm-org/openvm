use openvm_poseidon2_guest::{native_poseidon2_permute, POSEIDON2_STATE_BYTES, POSEIDON2_WIDTH};

/// Aligned wrapper for the Poseidon2 state buffer to satisfy the 4-byte alignment required by
/// `native_poseidon2_permute`.
#[repr(align(4))]
struct AlignedState([u8; POSEIDON2_STATE_BYTES]);

/// Applies the Poseidon2 permutation in place using the native zkvm instruction.
///
/// # Panics
///
/// Panics if any state word is not a canonical field element (i.e. is not less than
/// `0x78000001`).
pub fn permute(state: &mut [u32; POSEIDON2_WIDTH]) {
    assert!(
        state.iter().all(|&word| word < crate::BABY_BEAR_ORDER),
        "poseidon2 state words must be canonical field elements"
    );
    let mut buffer = AlignedState([0u8; POSEIDON2_STATE_BYTES]);
    for (i, &word) in state.iter().enumerate() {
        buffer.0[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
    }
    // SAFETY: buffer points to a POSEIDON2_STATE_BYTES-long, 4-byte-aligned buffer.
    unsafe { native_poseidon2_permute(buffer.0.as_mut_ptr()) };
    for (i, chunk) in buffer.0.chunks_exact(4).enumerate() {
        state[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }
}
