//! BLAKE3 utility functions and constants for the circuit extension.

use super::BLAKE3_BLOCK_BYTES;

// ============== BLAKE3 Constants ==============

/// BLAKE3 Initial Vector (IV) - same as SHA-256 but truncated to 8 words.
/// These are the first 32 bits of the fractional parts of the square roots of the first 8 primes.
pub const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// BLAKE3 message schedule (permutations for each round)
const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// BLAKE3 flag bits for compression function.
pub mod flags {
    /// First block of a chunk (1024 bytes max per chunk).
    pub const CHUNK_START: u32 = 1 << 0;
    /// Last block of a chunk.
    pub const CHUNK_END: u32 = 1 << 1;
    /// This compression is for the parent node in the tree.
    pub const PARENT: u32 = 1 << 2;
    /// This compression produces the final root output.
    pub const ROOT: u32 = 1 << 3;
    /// Input was derived from keyed hash.
    pub const KEYED_HASH: u32 = 1 << 4;
    /// Input was derived from derive_key context.
    pub const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
    /// Input was derived from derive_key material.
    pub const DERIVE_KEY_MATERIAL: u32 = 1 << 6;
}

// ============== BLAKE3 Compression Function ==============

/// The G mixing function used in BLAKE3 compression.
#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

/// One round of BLAKE3 compression.
#[inline(always)]
fn round(state: &mut [u32; 16], msg: &[u32; 16], round_idx: usize) {
    let schedule = MSG_SCHEDULE[round_idx];

    // Mix the columns
    g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
    g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
    g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
    g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

    // Mix the diagonals
    g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
    g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
    g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
    g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

/// BLAKE3 compression function.
///
/// This is the core function that processes a 64-byte block with an 8-word chaining value
/// and produces a new 8-word chaining value.
///
/// Similar to `keccakf` in Keccak, this function transforms the state in-place.
///
/// # Arguments
/// * `cv` - 8-word chaining value (mutated in place)
/// * `block` - 64-byte message block
/// * `block_len` - actual length of valid data in block (for padding)
/// * `counter` - block counter
/// * `flags` - BLAKE3 domain separation flags
pub fn blake3_compress(
    cv: &mut [u32; 8],
    block: &[u8; BLAKE3_BLOCK_BYTES],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    // Convert block bytes to 16 little-endian u32 words
    let mut msg = [0u32; 16];
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        msg[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }

    // Initialize state: [cv[0..8], IV[0..4], counter_lo, counter_hi, block_len, flags]
    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len as u32,
        flags as u32,
    ];

    // 7 rounds
    for i in 0..7 {
        round(&mut state, &msg, i);
    }

    // XOR first half with second half to produce new CV
    cv[0] = state[0] ^ state[8];
    cv[1] = state[1] ^ state[9];
    cv[2] = state[2] ^ state[10];
    cv[3] = state[3] ^ state[11];
    cv[4] = state[4] ^ state[12];
    cv[5] = state[5] ^ state[13];
    cv[6] = state[6] ^ state[14];
    cv[7] = state[7] ^ state[15];
}

// ============== Helper Functions ==============

/// Compute BLAKE3 hash of input bytes.
/// Returns the 32-byte digest.
#[inline]
pub fn blake3_hash(input: &[u8]) -> [u8; 32] {
    *blake3::hash(input).as_bytes()
}

/// Compute BLAKE3 hash using p3-blake3-air compatible parameters.
///
/// p3-blake3-air uses block_len=1, counter=0, flags=0 for all compressions,
/// which differs from standard BLAKE3. This function computes the same
/// output that p3-blake3-air would produce.
///
/// Note: This function zero-pads input to block boundaries. For trace
/// generation where actual memory values should be used, use `blake3_hash_p3_full_blocks`.
#[inline]
pub fn blake3_hash_p3(input: &[u8]) -> [u8; 32] {
    let num_blocks = num_blake3_compressions(input.len());
    let mut cv = BLAKE3_IV;

    // Pad input to full blocks
    let padded_len = num_blocks * BLAKE3_BLOCK_BYTES;
    let mut padded = vec![0u8; padded_len];
    padded[..input.len()].copy_from_slice(input);

    // Process each block with p3-compatible parameters
    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLAKE3_BLOCK_BYTES;
        let block: [u8; BLAKE3_BLOCK_BYTES] = padded[block_start..block_start + BLAKE3_BLOCK_BYTES]
            .try_into()
            .unwrap();

        // p3-blake3-air uses: block_len=1, counter=0, flags=0
        blake3_compress(&mut cv, &block, 1, 0, 0);
    }

    // Convert CV words to bytes
    let mut digest = [0u8; 32];
    for (i, word) in cv.iter().enumerate() {
        digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
    }
    digest
}

/// Compute BLAKE3 hash over full blocks using p3-blake3-air compatible parameters.
///
/// Unlike `blake3_hash_p3`, this function expects the input to already be
/// block-aligned (multiple of 64 bytes) and uses the actual input values
/// without any padding. This is used in trace generation where we hash
/// actual memory values.
#[inline]
pub fn blake3_hash_p3_full_blocks(full_blocks: &[u8]) -> [u8; 32] {
    assert!(
        full_blocks.len() % BLAKE3_BLOCK_BYTES == 0,
        "Input must be block-aligned"
    );
    let num_blocks = full_blocks.len() / BLAKE3_BLOCK_BYTES;
    let mut cv = BLAKE3_IV;

    // Process each block with p3-compatible parameters
    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLAKE3_BLOCK_BYTES;
        let block: [u8; BLAKE3_BLOCK_BYTES] = full_blocks
            [block_start..block_start + BLAKE3_BLOCK_BYTES]
            .try_into()
            .unwrap();

        // p3-blake3-air uses: block_len=1, counter=0, flags=0
        blake3_compress(&mut cv, &block, 1, 0, 0);
    }

    // Convert CV words to bytes
    let mut digest = [0u8; 32];
    for (i, word) in cv.iter().enumerate() {
        digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
    }
    digest
}

/// Number of compression function calls required for BLAKE3 on input of `byte_len` bytes.
///
/// BLAKE3 processes 64 bytes per compression (unlike chunks which are 1024 bytes).
/// For simple hashing (not tree mode), we need ceil(len / 64) compressions,
/// with at least 1 for empty input.
#[inline]
pub fn num_blake3_compressions(byte_len: usize) -> usize {
    if byte_len == 0 {
        1 // BLAKE3 always needs at least one compression for empty input
    } else {
        byte_len.div_ceil(BLAKE3_BLOCK_BYTES)
    }
}

/// Prepare the compression function input for p3-blake3-air.
///
/// The p3-blake3-air `generate_trace_rows` expects `Vec<[u32; 24]>` where each element is:
/// - [0..16]: 16 message words (64 bytes of input as little-endian u32s)
/// - [16..24]: 8 chaining value words
///
/// # Arguments
/// * `message_block` - 64 bytes of input (padded with zeros if partial)
/// * `chaining_value` - 8 u32 words of chaining value (IV for first block)
///
/// # Returns
/// The 24-word compression input array.
#[inline]
pub fn prepare_compression_input(
    message_block: &[u8; BLAKE3_BLOCK_BYTES],
    chaining_value: &[u32; 8],
) -> [u32; 24] {
    let mut result = [0u32; 24];

    // Pack message bytes into 16 little-endian u32 words
    for (i, chunk) in message_block.chunks_exact(4).enumerate() {
        result[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }

    // Copy chaining value
    result[16..24].copy_from_slice(chaining_value);

    result
}

/// Extract the compression output from p3-blake3-air's `outputs` field.
///
/// The `outputs` field in Blake3Cols is `[[[T; 32]; 4]; 4]` - 16 words stored as
/// 4 groups of 4 words, each word being 32 bits.
///
/// For the next compression's chaining value, we need the first 8 words XORed
/// with the second 8 words (standard BLAKE3 truncation).
///
/// This function converts bit-decomposed outputs back to u32 words.
#[inline]
pub fn bits_to_u32<F: Into<u64> + Copy>(bits: &[F; 32]) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0u32, |acc, (i, &bit)| acc | ((bit.into() as u32 & 1) << i))
}

/// Convert a u32 word to 32 bits (least significant bit first).
#[inline]
pub fn u32_to_bits(word: u32) -> [u8; 32] {
    core::array::from_fn(|i| ((word >> i) & 1) as u8)
}

/// Convert 4 bytes to 32 bits (for memory read → compression input conversion).
#[inline]
pub fn bytes_to_bits(bytes: &[u8; 4]) -> [u8; 32] {
    u32_to_bits(u32::from_le_bytes(*bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_blake3_compressions() {
        assert_eq!(num_blake3_compressions(0), 1);
        assert_eq!(num_blake3_compressions(1), 1);
        assert_eq!(num_blake3_compressions(63), 1);
        assert_eq!(num_blake3_compressions(64), 1);
        assert_eq!(num_blake3_compressions(65), 2);
        assert_eq!(num_blake3_compressions(128), 2);
        assert_eq!(num_blake3_compressions(129), 3);
    }

    #[test]
    fn test_bits_roundtrip() {
        let value = 0xDEADBEEF_u32;
        let bits = u32_to_bits(value);
        let recovered = bits_to_u32(&bits.map(|b| b as u64));
        assert_eq!(value, recovered);
    }

    #[test]
    fn test_prepare_compression_input() {
        let mut message = [0u8; 64];
        message[0..4].copy_from_slice(&0x12345678_u32.to_le_bytes());

        let input = prepare_compression_input(&message, &BLAKE3_IV);

        assert_eq!(input[0], 0x12345678);
        assert_eq!(input[16], BLAKE3_IV[0]);
        assert_eq!(input[23], BLAKE3_IV[7]);
    }

    /// Verify our blake3_compress matches p3-blake3-air outputs
    #[test]
    fn test_blake3_compress_matches_p3() {
        use openvm_stark_backend::p3_field::PrimeField64;
        use openvm_stark_sdk::p3_baby_bear::BabyBear;
        use p3_blake3_air::{generate_trace_rows, Blake3Cols, NUM_BLAKE3_COLS};
        use std::borrow::Borrow;

        type F = BabyBear;

        // Create a simple message block
        let mut message = [0u8; BLAKE3_BLOCK_BYTES];
        for i in 0..8 {
            message[i * 4..(i + 1) * 4].copy_from_slice(&(0x42424242_u32).to_le_bytes());
        }

        // Compute CV using our blake3_compress
        let mut our_cv = BLAKE3_IV;
        blake3_compress(&mut our_cv, &message, 1, 0, 0);

        // Generate p3 trace
        let compression_input = prepare_compression_input(&message, &BLAKE3_IV);
        let p3_trace = generate_trace_rows::<F>(vec![compression_input], 0);
        let row: &Blake3Cols<F> = p3_trace.values[..NUM_BLAKE3_COLS].borrow();

        // Extract p3's outputs[0..2] (first 8 words) - what our AIR constrains as next CV
        let mut p3_outputs_first_8 = [0u32; 8];
        for group in 0..2 {
            for word in 0..4 {
                let idx = group * 4 + word;
                p3_outputs_first_8[idx] =
                    bits_to_u32(&row.outputs[group][word].map(|f| f.as_canonical_u64()));
            }
        }

        // Our blake3_compress output should match p3's outputs[0..2]
        assert_eq!(
            our_cv, p3_outputs_first_8,
            "Our blake3_compress doesn't match p3-blake3-air outputs"
        );
    }
}
