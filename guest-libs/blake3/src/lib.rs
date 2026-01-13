//! OpenVM library for BLAKE3 hashing.
//!
//! This library provides BLAKE3 hashing for OpenVM guest programs.
//! Currently uses the pure Rust `blake3` crate implementation.
//! Once a native BLAKE3 extension is added, this will automatically
//! switch to the accelerated version when compiling for zkvm.

#![no_std]

/// The blake3 cryptographic hash function.
#[inline(always)]
pub fn blake3(input: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    set_blake3(input, &mut output);
    output
}

/// Sets `output` to the blake3 hash of `input`.
pub fn set_blake3(input: &[u8], output: &mut [u8; 32]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(input);
        output.copy_from_slice(hasher.finalize().as_bytes());
    }
    #[cfg(target_os = "zkvm")]
    openvm_blake3_guest::zkvm_blake3_impl(
        input.as_ptr(),
        input.len(),
        output.as_mut_ptr() as *mut u8,
    );
}
