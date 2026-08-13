#![no_std]

#[cfg(not(target_os = "zkvm"))]
mod host_impl;
#[cfg(target_os = "zkvm")]
mod zkvm_impl;

#[cfg(not(target_os = "zkvm"))]
pub use host_impl::permute;
pub use openvm_poseidon2_guest::{POSEIDON2_STATE_BYTES, POSEIDON2_WIDTH};
#[cfg(target_os = "zkvm")]
pub use zkvm_impl::permute;

/// Rate of the Poseidon2 sponge: number of field elements absorbed between permutations.
pub const POSEIDON2_RATE: usize = 8;
/// Size of the Poseidon2 digest, in field elements.
pub const DIGEST_SIZE: usize = 8;
/// The BabyBear field characteristic `2^31 - 2^27 + 1`.
const BABY_BEAR_ORDER: u32 = 0x7800_0001;
/// Bytes packed into one field element by [`hash_bytes`]. Three is the widest group that always
/// yields a canonical element, since `2^24 < BABY_BEAR_ORDER`.
pub const BYTES_PER_ELEMENT: usize = 3;

/// Poseidon2 sponge hasher over canonical field elements (each word must be less than the
/// BabyBear field characteristic `0x78000001`).
///
/// Construction:
/// - Width 16 state, all-zero initialized.
/// - Absorb: `state[idx] = word` (assign semantics); permute when `idx == POSEIDON2_RATE`.
/// - Finalize: write a padding marker `1` at the current index and zero-fill the rest of the rate
///   block (pad10*1), then one final permutation, and output the first `DIGEST_SIZE` words.
#[derive(Clone)]
pub struct Poseidon2 {
    state: [u32; POSEIDON2_WIDTH],
    idx: usize,
}

impl Poseidon2 {
    /// Creates a new Poseidon2 hasher in the zero state.
    pub const fn new() -> Self {
        Self {
            state: [0u32; POSEIDON2_WIDTH],
            idx: 0,
        }
    }

    /// Absorbs field elements into the sponge state.
    ///
    /// # Panics
    ///
    /// Panics if any input word is not a canonical field element (i.e. is not less than
    /// `0x78000001`).
    pub fn update(&mut self, input: &[u32]) {
        assert!(
            input.iter().all(|&word| word < BABY_BEAR_ORDER),
            "poseidon2 input words must be canonical field elements"
        );
        for &word in input {
            self.state[self.idx] = word;
            self.idx += 1;
            if self.idx == POSEIDON2_RATE {
                permute(&mut self.state);
                self.idx = 0;
            }
        }
    }

    /// Finalizes the hash and returns the `DIGEST_SIZE`-element digest.
    pub fn finalize(mut self) -> [u32; DIGEST_SIZE] {
        // pad10*1: write the marker and zero-fill the rest of the rate block so the final
        // permutation always processes a cleanly padded block.
        self.state[self.idx] = 1;
        self.state[self.idx + 1..POSEIDON2_RATE].fill(0);
        permute(&mut self.state);
        let mut digest = [0u32; DIGEST_SIZE];
        digest.copy_from_slice(&self.state[..DIGEST_SIZE]);
        digest
    }
}

impl Default for Poseidon2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Hashes a slice of canonical field elements.
pub fn hash_u32s(input: &[u32]) -> [u32; DIGEST_SIZE] {
    let mut hasher = Poseidon2::new();
    hasher.update(input);
    hasher.finalize()
}

/// Hashes an arbitrary byte string.
///
/// Accepts any input: every length, every byte value. The Poseidon2 permutation operates on
/// BabyBear elements, which cannot hold a full 32-bit word, so bytes are packed
/// [`BYTES_PER_ELEMENT`] at a time into little-endian field elements. Three bytes is the widest
/// group that always fits (`2^24 < 0x78000001`), so no input can produce a non-canonical element.
///
/// The byte string is first padded with a `0x01` marker followed by zeroes up to a multiple of
/// [`BYTES_PER_ELEMENT`] (a byte-level `pad10*1`). Without it, inputs differing only in trailing
/// zeroes — `[0x07]` and `[0x07, 0x00]` — would pack to the same field elements and collide. The
/// padding is always applied, including when the length is already a multiple of 3, so the encoding
/// stays injective.
///
/// Note this is *not* interchangeable with [`hash_u32s`]: that one absorbs 4-byte words directly
/// and requires each to be a canonical field element, so the two disagree on the same bytes.
pub fn hash_bytes(input: &[u8]) -> [u8; DIGEST_SIZE * 4] {
    let mut hasher = Poseidon2::new();
    let mut groups = input.chunks_exact(BYTES_PER_ELEMENT);
    for group in &mut groups {
        hasher.update(&[pack_le(&[group[0], group[1], group[2]])]);
    }

    // pad10*1 over the trailing partial group. `remainder()` is shorter than BYTES_PER_ELEMENT, so
    // there is always room for the marker byte.
    let remainder = groups.remainder();
    let mut tail = [0u8; BYTES_PER_ELEMENT];
    tail[..remainder.len()].copy_from_slice(remainder);
    tail[remainder.len()] = 1;
    hasher.update(&[pack_le(&tail)]);

    let digest = hasher.finalize();
    let mut output = [0u8; DIGEST_SIZE * 4];
    for (i, word) in digest.iter().enumerate() {
        output[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
    }
    output
}

/// Packs a [`BYTES_PER_ELEMENT`]-byte group into a little-endian field element. The result is less
/// than `2^24`, hence always canonical.
fn pack_le(group: &[u8; BYTES_PER_ELEMENT]) -> u32 {
    u32::from_le_bytes([group[0], group[1], group[2], 0])
}
