//! OpenVM library for BLAKE3 hashing.
//!
//! This library provides BLAKE3 hashing for OpenVM guest programs.
//! Currently uses the pure Rust `blake3` crate implementation.
//! Once a native BLAKE3 extension is added, this will automatically
//! switch to the accelerated version when compiling for zkvm.

#![no_std]

extern crate alloc;

/// The blake3 cryptographic hash function.
/// Returns a 32-byte hash of the input.
#[inline(always)]
pub fn blake3(input: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    set_blake3(input, &mut output);
    output
}

/// Sets `output` to the blake3 hash of `input`.
#[inline(always)]
pub fn set_blake3(input: &[u8], output: &mut [u8; 32]) {
    // For now, use pure Rust implementation for both native and zkvm
    // This will be replaced with native extension call once available:
    //
    // #[cfg(target_os = "zkvm")]
    // {
    //     openvm_blake3_guest::native_blake3(
    //         input.as_ptr(),
    //         input.len(),
    //         output.as_mut_ptr(),
    //     );
    // }
    // #[cfg(not(target_os = "zkvm"))]
    
    let hash = blake3::hash(input);
    output.copy_from_slice(hash.as_bytes());
}

/// Create a BLAKE3 hasher for incremental hashing.
/// Useful for hashing large data in chunks.
pub fn hasher() -> blake3::Hasher {
    blake3::Hasher::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_empty() {
        let hash = blake3(b"");
        let expected = hex::decode(
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        ).unwrap();
        assert_eq!(hash.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_blake3_hello_world() {
        let hash = blake3(b"hello world");
        let expected = hex::decode(
            "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24"
        ).unwrap();
        assert_eq!(hash.as_slice(), expected.as_slice());
    }
}
