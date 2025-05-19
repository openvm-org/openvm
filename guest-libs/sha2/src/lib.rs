#![no_std]

/// The sha256 cryptographic hash function.
#[inline(always)]
pub fn sha256(input: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    set_sha256(input, &mut output);
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
        openvm_sha256_guest::zkvm_sha256_impl(
            input.as_ptr(),
            input.len(),
            output.as_mut_ptr() as *mut u8,
        );
    }
}
