use super::KECCAK_WIDTH_BYTES;

/// XOR input bytes into state starting at the given offset.
#[inline(always)]
pub fn xorin(state: &mut [u8; KECCAK_WIDTH_BYTES], offset: usize, input: &[u8]) {
    openvm_keccak256_guest::native_xorin(state[offset..].as_mut_ptr(), input.as_ptr(), input.len());
}

/// Keccak-f[1600] permutation using native zkvm instruction
#[inline(always)]
pub fn keccakf(state: &mut [u8; KECCAK_WIDTH_BYTES]) {
    openvm_keccak256_guest::native_keccakf(state.as_mut_ptr());
}
