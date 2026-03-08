use core::cmp::min;

use openvm_keccak256_guest::{KECCAK_OUTPUT_SIZE, KECCAK_RATE, KECCAK_WIDTH_BYTES};
use spin::Mutex;

/// Aligned wrapper for Keccak state to ensure 4-byte alignment on word accesses.
#[derive(Clone)]
#[repr(align(4))]
struct AlignedState([u8; KECCAK_WIDTH_BYTES]);

/// Keccak-256 hasher state for incremental hashing.
///
/// This struct implements the Keccak sponge construction for Keccak-256.
#[derive(Clone)]
pub struct Keccak256 {
    /// The Keccak state (1600 bits = 200 bytes)
    state: AlignedState,
    /// Current position in the rate portion of the state (0 <= idx < KECCAK_RATE)
    idx: usize,
}

impl Keccak256 {
    /// Creates a new Keccak-256 hasher.
    pub const fn new() -> Self {
        Self {
            state: AlignedState([0u8; KECCAK_WIDTH_BYTES]),
            idx: 0,
        }
    }

    /// Resets the hasher to its initial state.
    fn reset(&mut self) {
        self.state.0.fill(0);
        self.idx = 0;
    }

    /// XOR input bytes into state at the current index and advance the index.
    #[inline(always)]
    fn xorin(&mut self, input: *const u8, len: usize) {
        openvm_keccak256_guest::native_xorin(self.state.0[self.idx..].as_mut_ptr(), input, len);
        self.idx += len;
    }

    /// Keccak-f[1600] permutation using native zkvm instruction.
    #[inline(always)]
    fn keccakf(&mut self) {
        openvm_keccak256_guest::native_keccakf(self.state.0.as_mut_ptr());
    }

    /// Absorbs input data into the sponge state from a raw pointer.
    fn update_ptr(&mut self, mut input: *const u8, mut len: usize) {
        while len > 0 {
            let to_absorb = min(len, KECCAK_RATE - self.idx);

            // XOR input into state
            self.xorin(input, to_absorb);

            // If we filled the rate portion, apply the permutation
            if self.idx == KECCAK_RATE {
                self.keccakf();
                self.idx = 0;
            }

            input = unsafe { input.add(to_absorb) };
            len -= to_absorb;
        }
    }

    /// Absorbs input data into the sponge state.
    #[inline(always)]
    pub fn update(&mut self, input: &[u8]) {
        self.update_ptr(input.as_ptr(), input.len());
    }

    /// Finalizes the hash computation and writes the result to a raw pointer.
    fn finalize_ptr(&mut self, output: *mut u8) {
        // Apply Keccak padding (pad10*1): 0x01 at current position, 0x80 at end of rate
        self.state.0[self.idx] ^= 0x01;
        self.state.0[KECCAK_RATE - 1] ^= 0x80;

        // Final permutation
        self.keccakf();

        // Squeeze: copy output from state
        unsafe {
            core::ptr::copy_nonoverlapping(self.state.0.as_ptr(), output, KECCAK_OUTPUT_SIZE);
        }
    }

    /// Finalizes the hash computation and writes the result to the output buffer.
    ///
    /// The output buffer must be at least `KECCAK_OUTPUT_SIZE` (32) bytes.
    #[inline(always)]
    pub fn finalize(mut self, output: &mut [u8]) {
        debug_assert!(
            output.len() >= KECCAK_OUTPUT_SIZE,
            "output buffer too small"
        );
        self.finalize_ptr(output.as_mut_ptr());
    }
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Static hasher for `native_keccak256`. Reusing a static buffer avoids stack frame growth.
/// Spin mutex is acceptable since the zkVM is single-threaded so there is no contention.
static KECCAK256_HASHER: Mutex<Keccak256> = Mutex::new(Keccak256::new());

/// Native hook for keccak256 for use with `alloy-primitives` "native-keccak" feature.
///
/// # Safety
///
/// The VM accepts the preimage by pointer and length, and writes the
/// 32-byte hash.
/// - `bytes` must point to an input buffer at least `len` long.
/// - `output` must point to a buffer that is at least 32-bytes long.
///
/// [`keccak256`]: https://en.wikipedia.org/wiki/SHA-3
/// [`sha3`]: https://docs.rs/sha3/latest/sha3/
/// [`tiny_keccak`]: https://docs.rs/tiny-keccak/latest/tiny_keccak/
#[no_mangle]
pub extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    let mut hasher = KECCAK256_HASHER.lock();
    hasher.update_ptr(bytes, len);
    hasher.finalize_ptr(output);
    hasher.reset();
}

/// Sets `output` to the keccak256 hash of `input`.
#[inline(always)]
pub fn set_keccak256(input: &[u8], output: &mut [u8; KECCAK_OUTPUT_SIZE]) {
    native_keccak256(input.as_ptr(), input.len(), output.as_mut_ptr());
}
