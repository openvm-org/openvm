#![no_std]

/// The sha256 cryptographic hash function.
#[inline(always)]
pub fn sha256(input: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    set_sha256(input, &mut output);
    output
}

/// The sha512 cryptographic hash function.
#[inline(always)]
pub fn sha512(input: &[u8]) -> [u8; 64] {
    let mut output = [0u8; 64];
    set_sha512(input, &mut output);
    output
}

/// The sha384 cryptographic hash function.
#[inline(always)]
pub fn sha384(input: &[u8]) -> [u8; 48] {
    let mut output = [0u8; 48];
    set_sha384(input, &mut output);
    output
}

/// Sets `output` to the sha256 hash of `input`.
pub fn set_sha256(input: &[u8], output: &mut [u8; 32]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(input);
        output.copy_from_slice(hasher.finalize().as_ref());
    }
    #[cfg(target_os = "zkvm")]
    {
        openvm_sha2_guest::zkvm_sha256_impl(
            input.as_ptr(),
            input.len(),
            output.as_mut_ptr() as *mut u8,
        );
    }
}

/// Sets `output` to the sha512 hash of `input`.
pub fn set_sha512(input: &[u8], output: &mut [u8; 64]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use sha2::{Digest, Sha512};
        let mut hasher = Sha512::new();
        hasher.update(input);
        output.copy_from_slice(hasher.finalize().as_ref());
    }
    #[cfg(target_os = "zkvm")]
    {
        openvm_sha2_guest::zkvm_sha512_impl(
            input.as_ptr(),
            input.len(),
            output.as_mut_ptr() as *mut u8,
        );
    }
}

/// Sets the first 48 bytes of `output` to the sha384 hash of `input`.
/// Sets the last 16 bytes to zeros.
pub fn set_sha384(input: &[u8], output: &mut [u8; 48]) {
    #[cfg(not(target_os = "zkvm"))]
    {
        use sha2::{Digest, Sha384};
        let mut hasher = Sha384::new();
        hasher.update(input);
        output.copy_from_slice(hasher.finalize().as_ref());
    }
    #[cfg(target_os = "zkvm")]
    {
        let mut output_64: [u8; 64] = [0; 64];
        openvm_sha2_guest::zkvm_sha384_impl(
            input.as_ptr(),
            input.len(),
            output_64.as_mut_ptr() as *mut u8,
        );
        output.copy_from_slice(&output_64[..48]);
    }
}
