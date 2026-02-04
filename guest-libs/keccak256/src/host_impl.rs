use openvm_keccak256_guest::KECCAK_OUTPUT_SIZE;
use tiny_keccak::Hasher as _;

/// Keccak-256 hasher state for incremental hashing.
///
/// This struct wraps tiny-keccak's Keccak hasher for Keccak-256.
#[derive(Clone)]
pub struct Keccak256 {
    inner: tiny_keccak::Keccak,
}

impl Keccak256 {
    /// Creates a new Keccak-256 hasher.
    pub fn new() -> Self {
        Self {
            inner: tiny_keccak::Keccak::v256(),
        }
    }

    /// Absorbs input data into the sponge state.
    pub fn update(&mut self, input: &[u8]) {
        self.inner.update(input);
    }

    /// Finalizes the hash computation and writes the result to the output buffer.
    ///
    /// The output buffer must be at least `KECCAK_OUTPUT_SIZE` (32) bytes.
    pub fn finalize(self, output: &mut [u8]) {
        debug_assert!(
            output.len() >= super::KECCAK_OUTPUT_SIZE,
            "output buffer too small"
        );
        self.inner.finalize(output);
    }
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Sets `output` to the keccak256 hash of `input`.
#[inline(always)]
pub fn set_keccak256(input: &[u8], output: &mut [u8; KECCAK_OUTPUT_SIZE]) {
    let mut hasher = Keccak256::new();
    hasher.update(input);
    hasher.finalize(output);
}
