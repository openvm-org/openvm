use super::KECCAK_WIDTH_BYTES;

/// Number of u64 words in Keccak state (200 bytes / 8 bytes per u64 = 25)
const KECCAK_STATE_WORDS: usize = KECCAK_WIDTH_BYTES / 8;

/// Number of Keccak-f rounds
const KECCAK_ROUNDS: usize = 24;

/// Keccak state dimension (5x5 matrix of u64 words)
const KECCAK_DIM: usize = 5;

// Round constants for Keccak-f[1600]
const RC: [u64; KECCAK_ROUNDS] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

// Rotation offsets for rho step
const RHO: [u32; KECCAK_ROUNDS] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

// Permutation indices for pi step
const PI: [usize; KECCAK_ROUNDS] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

/// XOR input bytes into state starting at the given offset.
#[inline(always)]
pub fn xorin(state: &mut [u8; KECCAK_WIDTH_BYTES], offset: usize, input: &[u8]) {
    for (i, byte) in input.iter().enumerate() {
        state[offset + i] ^= byte;
    }
}

/// Keccak-f[1600] permutation on byte state
#[inline(always)]
pub fn keccakf(state: &mut [u8; KECCAK_WIDTH_BYTES]) {
    // Convert bytes to u64 array (little-endian)
    let mut state_u64 = [0u64; KECCAK_STATE_WORDS];
    for (i, chunk) in state.chunks_exact(8).enumerate() {
        state_u64[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }

    // Apply permutation
    let mut bc = [0u64; KECCAK_DIM];

    for rc in &RC {
        // Theta step
        for i in 0..KECCAK_DIM {
            bc[i] = state_u64[i]
                ^ state_u64[i + KECCAK_DIM]
                ^ state_u64[i + 2 * KECCAK_DIM]
                ^ state_u64[i + 3 * KECCAK_DIM]
                ^ state_u64[i + 4 * KECCAK_DIM];
        }
        for i in 0..KECCAK_DIM {
            let t = bc[(i + 4) % KECCAK_DIM] ^ bc[(i + 1) % KECCAK_DIM].rotate_left(1);
            for j in (0..KECCAK_STATE_WORDS).step_by(KECCAK_DIM) {
                state_u64[j + i] ^= t;
            }
        }

        // Rho and Pi steps
        let mut t = state_u64[1];
        for i in 0..KECCAK_ROUNDS {
            let j = PI[i];
            let temp = state_u64[j];
            state_u64[j] = t.rotate_left(RHO[i]);
            t = temp;
        }

        // Chi step
        for j in (0..KECCAK_STATE_WORDS).step_by(KECCAK_DIM) {
            bc.copy_from_slice(&state_u64[j..j + KECCAK_DIM]);
            for i in 0..KECCAK_DIM {
                state_u64[j + i] ^= (!bc[(i + 1) % KECCAK_DIM]) & bc[(i + 2) % KECCAK_DIM];
            }
        }

        // Iota step
        state_u64[0] ^= rc;
    }

    // Convert back to bytes
    for (i, word) in state_u64.iter().enumerate() {
        state[i * 8..(i + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
}
