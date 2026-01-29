use core::cmp::min;

use sha2::digest::{
    consts::{U32, U48, U64},
    FixedOutput, HashMarker, Output, OutputSizeUser, Update,
};

// We store static padding bytes here so that we don't need to allocate a vector when padding in
// finalize().
// Padding always consists of a single 0x80 byte, followed by zeros, (and then the length of the
// message in bits but we don't include that here because it's not static).
// Length of this array is chosen to be the maximum block size between SHA-256 and SHA-512, since
// padding can be at most BLOCK_BYTES bytes.
const PADDING_BYTES: [u8; SHA512_BLOCK_BYTES] = {
    let mut padding_bytes = [0u8; SHA512_BLOCK_BYTES];
    padding_bytes[0] = 0x80;
    padding_bytes
};

const SHA256_STATE_WORDS: usize = 8;
const SHA256_BLOCK_BYTES: usize = 64;
const SHA256_DIGEST_BYTES: usize = 32;

// Initial state for SHA-256 in 32-bit words
const SHA256_H: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[derive(Debug, Clone, Copy)]
pub struct Sha256 {
    // the current hasher state, in 32-bit words
    state: [u32; SHA256_STATE_WORDS],
    // the next block of input
    buffer: [u8; SHA256_BLOCK_BYTES],
    // idx of next byte to write to buffer (equal to len mod SHA256_BLOCK_BYTES)
    idx: usize,
    // accumulated length of the input data, in bytes
    len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: SHA256_H,
            buffer: [0; SHA256_BLOCK_BYTES],
            idx: 0,
            len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.len += input.len() as u64;
        while !input.is_empty() {
            let to_copy = min(input.len(), SHA256_BLOCK_BYTES - self.idx);
            self.buffer[self.idx..self.idx + to_copy].copy_from_slice(&input[..to_copy]);
            self.idx += to_copy;
            if self.idx == SHA256_BLOCK_BYTES {
                self.idx = 0;
                self.compress();
            }
            input = &input[to_copy..];
        }
    }

    fn finalize(mut self) -> [u8; SHA256_DIGEST_BYTES] {
        // pad positive amount so that length is multiple of SHA256_BLOCK_BYTES
        // (extra 8 bytes are for message length)
        let num_bytes_of_padding = SHA256_BLOCK_BYTES - (self.idx + 8) % SHA256_BLOCK_BYTES;
        let message_len_in_bits: u64 = self.len * 8;
        self.update(&PADDING_BYTES[..num_bytes_of_padding]);
        self.update(&(message_len_in_bits).to_be_bytes());
        let mut output = [0u8; SHA256_DIGEST_BYTES];
        output
            .chunks_exact_mut(4)
            .zip(self.state.iter())
            .for_each(|(chunk, x)| {
                chunk.copy_from_slice(&x.to_be_bytes());
            });
        output
    }

    fn compress(&mut self) {
        openvm_sha2_guest::zkvm_sha256_impl(
            self.state.as_ptr() as *const u8,
            self.buffer.as_ptr(),
            self.state.as_mut_ptr() as *mut u8,
        );
    }
}

// We will implement FixedOutput, Default, Update, and HashMarker for Sha256 so that
// the blanket implementation of sha2::Digest is available.
// See: https://docs.rs/sha2/latest/sha2/trait.Digest.html#impl-Digest-for-D
impl Update for Sha256 {
    fn update(&mut self, input: &[u8]) {
        self.update(input);
    }
}

// OutputSizeUser is required for FixedOutput
// See: https://docs.rs/digest/0.10.7/digest/trait.FixedOutput.html
impl OutputSizeUser for Sha256 {
    type OutputSize = U32;
}

impl FixedOutput for Sha256 {
    fn finalize_into(self, out: &mut Output<Self>) {
        out.copy_from_slice(&self.finalize());
    }
}

impl HashMarker for Sha256 {}

const SHA512_STATE_WORDS: usize = 8;
const SHA512_BLOCK_BYTES: usize = 128;
const SHA512_DIGEST_BYTES: usize = 64;

// Initial state for SHA-512 in 64-bit words
pub const SHA512_H: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

#[derive(Debug, Clone, Copy)]
pub struct Sha512 {
    // the current hasher state
    state: [u64; SHA512_STATE_WORDS],
    // the next block of input
    buffer: [u8; SHA512_BLOCK_BYTES],
    // idx of next byte to write to buffer
    idx: usize,
    // accumulated length of the input data, in bytes
    len: u128,
}

impl Default for Sha512 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha512 {
    pub fn new() -> Self {
        Self {
            state: SHA512_H,
            buffer: [0; SHA512_BLOCK_BYTES],
            idx: 0,
            len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.len += input.len() as u128;
        while !input.is_empty() {
            let to_copy = min(input.len(), SHA512_BLOCK_BYTES - self.idx);
            self.buffer[self.idx..self.idx + to_copy].copy_from_slice(&input[..to_copy]);
            self.idx += to_copy;
            if self.idx == SHA512_BLOCK_BYTES {
                self.idx = 0;
                self.compress();
            }
            input = &input[to_copy..];
        }
    }

    fn finalize(mut self) -> [u8; SHA512_DIGEST_BYTES] {
        // pad positive amount so that length is multiple of SHA512_BLOCK_BYTES
        // (extra 16 bytes are for message length)
        let num_bytes_of_padding = SHA512_BLOCK_BYTES - (self.idx + 16) % SHA512_BLOCK_BYTES;
        let message_len_in_bits: u128 = self.len * 8;
        self.update(&PADDING_BYTES[..num_bytes_of_padding]);
        self.update(&(message_len_in_bits).to_be_bytes());
        let mut output = [0u8; SHA512_DIGEST_BYTES];
        output
            .chunks_exact_mut(8)
            .zip(self.state.iter())
            .for_each(|(chunk, x)| {
                chunk.copy_from_slice(&x.to_be_bytes());
            });
        output
    }

    fn compress(&mut self) {
        openvm_sha2_guest::zkvm_sha512_impl(
            self.state.as_ptr() as *const u8,
            self.buffer.as_ptr(),
            self.state.as_mut_ptr() as *mut u8,
        );
    }
}

// We will implement FixedOutput, Default, Update, and HashMarker for Sha512 so that
// the blanket implementation of sha2::Digest is available.
// See: https://docs.rs/sha2/latest/sha2/trait.Digest.html#impl-Digest-for-D
impl Update for Sha512 {
    fn update(&mut self, input: &[u8]) {
        self.update(input);
    }
}

// OutputSizeUser is required for FixedOutput
// See: https://docs.rs/digest/0.10.7/digest/trait.FixedOutput.html
impl OutputSizeUser for Sha512 {
    type OutputSize = U64;
}

impl FixedOutput for Sha512 {
    fn finalize_into(self, out: &mut Output<Self>) {
        out.copy_from_slice(&self.finalize());
    }
}

impl HashMarker for Sha512 {}

const SHA384_DIGEST_BYTES: usize = 48;

// Initial state for SHA-384 in 64-bit words
pub const SHA384_H: [u64; 8] = [
    0xcbbb9d5dc1059ed8,
    0x629a292a367cd507,
    0x9159015a3070dd17,
    0x152fecd8f70e5939,
    0x67332667ffc00b31,
    0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7,
    0x47b5481dbefa4fa4,
];

#[derive(Debug, Clone, Copy)]
pub struct Sha384 {
    inner: Sha512,
}

impl Default for Sha384 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha384 {
    pub fn new() -> Self {
        let mut inner = Sha512::new();
        inner.state = SHA384_H;
        Self { inner }
    }

    pub fn update(&mut self, input: &[u8]) {
        self.inner.update(input);
    }

    pub fn finalize(self) -> [u8; SHA384_DIGEST_BYTES] {
        let digest = self.inner.finalize();
        digest[..SHA384_DIGEST_BYTES].try_into().unwrap()
    }
}

// We will implement FixedOutput, Default, Update, and HashMarker for Sha384 so that
// the blanket implementation of sha2::Digest is available.
// See: https://docs.rs/sha2/latest/sha2/trait.Digest.html#impl-Digest-for-D
impl Update for Sha384 {
    fn update(&mut self, input: &[u8]) {
        self.update(input);
    }
}

// OutputSizeUser is required for FixedOutput
// See: https://docs.rs/digest/0.10.7/digest/trait.FixedOutput.html
impl OutputSizeUser for Sha384 {
    type OutputSize = U48;
}

impl FixedOutput for Sha384 {
    fn finalize_into(self, out: &mut Output<Self>) {
        out.copy_from_slice(&self.finalize());
    }
}

impl HashMarker for Sha384 {}
