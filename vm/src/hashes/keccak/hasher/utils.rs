use tiny_keccak::{keccakf, Hasher, Keccak};

use super::KECCAK_RATE_BYTES;

/// Wrapper function for tiny-keccak's keccak-f permutation.
/// Returns the new state after permutation.
pub fn keccak_f(mut state: [u64; 25]) -> [u64; 25] {
    keccakf(&mut state);
    state
}

pub fn keccak256(input: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(input);
    let mut output = [0u8; 32];
    hasher.finalize(&mut output);
    output
}

/// Number of keccak-f permutations required for keccak256 on
/// input of `byte_len` bytes.
pub fn num_keccak_f(byte_len: usize) -> usize {
    // always need at least 1 extra byte for padding
    // ceil((byte_len + 1) / rate) = byte_len // rate + 1
    byte_len / KECCAK_RATE_BYTES + 1
}
