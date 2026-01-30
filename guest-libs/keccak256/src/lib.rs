#![no_std]

use core::cmp::min;

#[cfg(not(target_os = "zkvm"))]
mod host_impl;
#[cfg(target_os = "zkvm")]
mod zkvm_impl;

#[cfg(not(target_os = "zkvm"))]
use host_impl::{keccakf, xorin};
#[cfg(target_os = "zkvm")]
use zkvm_impl::{keccakf, xorin};

/// Keccak state width in bytes (1600 bits = 200 bytes)
pub const KECCAK_WIDTH_BYTES: usize = 200;

/// Output size for Keccak-256 in bytes
pub const KECCAK_OUTPUT_SIZE: usize = 32;

/// Rate for Keccak-256 in bytes (capacity = 512 bits, rate = 1600 - 512 = 1088 bits = 136 bytes)
const KECCAK256_RATE: usize = 136;

/// Keccak-256 hasher state for incremental hashing.
///
/// This struct implements the Keccak sponge construction for Keccak-256.
#[derive(Clone)]
#[repr(align(4))]
pub struct Keccak256 {
    /// The Keccak state (1600 bits = 200 bytes)
    state: [u8; KECCAK_WIDTH_BYTES],
    /// Current position in the rate portion of the state (0 <= idx < KECCAK256_RATE)
    idx: usize,
}

impl Keccak256 {
    /// Creates a new Keccak-256 hasher.
    pub fn new() -> Self {
        Self {
            state: [0u8; KECCAK_WIDTH_BYTES],
            idx: 0,
        }
    }

    /// Absorbs input data into the sponge state.
    pub fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            let to_absorb = min(input.len(), KECCAK256_RATE - self.idx);

            // XOR input into state
            xorin(&mut self.state, self.idx, &input[..to_absorb]);
            self.idx += to_absorb;

            // If we filled the rate portion, apply the permutation
            if self.idx == KECCAK256_RATE {
                keccakf(&mut self.state);
                self.idx = 0;
            }

            input = &input[to_absorb..];
        }
    }

    /// Finalizes the hash computation and writes the result to the output buffer.
    ///
    /// The output buffer must be at least `KECCAK_OUTPUT_SIZE` (32) bytes.
    pub fn finalize(mut self, output: &mut [u8]) {
        debug_assert!(
            output.len() >= KECCAK_OUTPUT_SIZE,
            "output buffer too small"
        );

        // Apply Keccak padding (pad10*1): 0x01 at current position, 0x80 at end of rate
        self.state[self.idx] ^= 0x01;
        self.state[KECCAK256_RATE - 1] ^= 0x80;

        // Final permutation
        keccakf(&mut self.state);

        // Squeeze: copy output from state
        output[..KECCAK_OUTPUT_SIZE].copy_from_slice(&self.state[..KECCAK_OUTPUT_SIZE]);
    }
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "tiny_keccak")]
impl tiny_keccak::Hasher for Keccak256 {
    fn update(&mut self, input: &[u8]) {
        Keccak256::update(self, input);
    }

    fn finalize(self, output: &mut [u8]) {
        Keccak256::finalize(self, output);
    }
}

/// Computes the keccak256 hash of the input.
#[inline(always)]
pub fn keccak256(input: &[u8]) -> [u8; KECCAK_OUTPUT_SIZE] {
    let mut output = [0u8; KECCAK_OUTPUT_SIZE];
    set_keccak256(input, &mut output);
    output
}

/// Sets `output` to the keccak256 hash of `input`.
#[inline(always)]
pub fn set_keccak256(input: &[u8], output: &mut [u8; KECCAK_OUTPUT_SIZE]) {
    let mut hasher = Keccak256::new();
    hasher.update(input);
    hasher.finalize(output);
}
