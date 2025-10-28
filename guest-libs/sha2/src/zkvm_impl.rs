use core::cmp::min;

use sha2::digest::{
    consts::{U32, U64},
    FixedOutput, HashMarker, Output, OutputSizeUser, Update,
};

// TODO: the three implementations can be merged into one using a macro

// We store static padding bytes so that we don't need to allocate a vector when padding in
// finalize().
// Padding always consists of a single 0x80 byte, followed by zeros, (and then the length of the
// message in bits but we don't include that here because it's not static).
// Length is chosen to be the maximum block size between SHA-256 and SHA-512, since padding can be
// at most BLOCK_BYTES bytes.
const PADDING_BYTES: [u8; SHA512_BLOCK_BYTES] = {
    let mut padding_bytes = [0u8; SHA512_BLOCK_BYTES];
    padding_bytes[0] = 0x80;
    padding_bytes
};

const SHA256_STATE_BYTES: usize = 32;
const SHA256_BLOCK_BYTES: usize = 64;
const SHA256_DIGEST_BYTES: usize = 32;

// Initial state for SHA-256 in big-endian bytes
const SHA256_H: [u8; SHA256_STATE_BYTES] = [
    106, 9, 230, 103, 187, 103, 174, 133, 60, 110, 243, 114, 165, 79, 245, 58, 81, 14, 82, 127,
    155, 5, 104, 140, 31, 131, 217, 171, 91, 224, 205, 25,
];

#[derive(Debug, Clone, Copy)]
pub struct Sha256 {
    // the current hasher state, in big-endian
    state: [u8; SHA256_STATE_BYTES],
    // the next block of input
    buffer: [u8; SHA256_BLOCK_BYTES],
    // idx of next byte to write to buffer (equal to len mod SHA256_BLOCK_BYTES)
    idx: usize,
    // accumulated length of the input data, in bytes
    len: usize,
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

    pub fn update(&mut self, mut input: &[u8]) {
        self.len += input.len();
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

    pub fn finalize(mut self) -> [u8; SHA256_DIGEST_BYTES] {
        // pad until length in bytes is 56 mod 64 (leave 8 bytes for the message length)
        let num_bytes_of_padding = SHA256_BLOCK_BYTES - 8 - self.idx;
        // ensure num_bytes_of_padding is positive
        let num_bytes_of_padding = (num_bytes_of_padding + SHA256_BLOCK_BYTES) % SHA256_BLOCK_BYTES;
        let message_len_in_bits = self.len * 8;
        self.update(&PADDING_BYTES[..num_bytes_of_padding]);
        self.update(&(message_len_in_bits as u64).to_be_bytes());
        self.state
    }

    fn compress(&mut self) {
        openvm_sha2_guest::zkvm_sha256_impl(
            self.state.as_ptr(),
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

const SHA512_STATE_BYTES: usize = 64;
const SHA512_BLOCK_BYTES: usize = 128;
const SHA512_DIGEST_BYTES: usize = 64;

// Initial state for SHA-512 in big-endian bytes
const SHA512_H: [u8; SHA512_STATE_BYTES] = [
    106, 9, 230, 103, 243, 188, 201, 8, 187, 103, 174, 133, 132, 202, 167, 59, 60, 110, 243, 114,
    254, 148, 248, 43, 165, 79, 245, 58, 95, 29, 54, 241, 81, 14, 82, 127, 173, 230, 130, 209, 155,
    5, 104, 140, 43, 62, 108, 31, 31, 131, 217, 171, 251, 65, 189, 107, 91, 224, 205, 25, 19, 126,
    33, 121,
];

#[derive(Debug, Clone, Copy)]
pub struct Sha512 {
    // the current hasher state
    state: [u8; SHA512_STATE_BYTES],
    // the next block of input
    buffer: [u8; SHA512_BLOCK_BYTES],
    // idx of next byte to write to buffer
    idx: usize,
    // accumulated length of the input data, in bytes
    len: usize,
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

    pub fn update(&mut self, mut input: &[u8]) {
        self.len += input.len();
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

    pub fn finalize(mut self) -> [u8; SHA512_DIGEST_BYTES] {
        // pad until length in bytes is 112 mod 128 (leave 16 bytes for the message length)
        let num_bytes_of_padding = SHA512_BLOCK_BYTES - 16 - self.idx;
        // ensure num_bytes_of_padding is positive
        let num_bytes_of_padding = (num_bytes_of_padding + SHA512_BLOCK_BYTES) % SHA512_BLOCK_BYTES;
        let message_len_in_bits = self.len * 8;
        self.update(&PADDING_BYTES[..num_bytes_of_padding]);
        self.update(&(message_len_in_bits as u128).to_be_bytes());
        self.state
    }

    fn compress(&mut self) {
        openvm_sha2_guest::zkvm_sha512_impl(
            self.state.as_ptr(),
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

const SHA384_STATE_BYTES: usize = 64;
const SHA384_BLOCK_BYTES: usize = 128;
const SHA384_DIGEST_BYTES: usize = 48;

const SHA384_H: [u8; SHA384_STATE_BYTES] = [
    203, 187, 157, 93, 193, 5, 158, 216, 98, 154, 41, 42, 54, 124, 213, 7, 145, 89, 1, 90, 48, 112,
    221, 23, 21, 47, 236, 216, 247, 14, 89, 57, 103, 51, 38, 103, 255, 192, 11, 49, 142, 180, 74,
    135, 104, 88, 21, 17, 219, 12, 46, 13, 100, 249, 143, 167, 71, 181, 72, 29, 190, 250, 79, 164,
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

    fn compress(&mut self) {
        self.inner.compress();
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
    type OutputSize = U64;
}

impl FixedOutput for Sha384 {
    fn finalize_into(self, out: &mut Output<Self>) {
        out.copy_from_slice(&self.finalize());
    }
}

impl HashMarker for Sha384 {}
