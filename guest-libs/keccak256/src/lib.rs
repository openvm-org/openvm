#![no_std]

use openvm_keccak256_guest::KECCAK_OUTPUT_SIZE;

#[cfg(not(any(target_os = "zkvm", feature = "tiny_keccak")))]
compile_error!("Keccak256 requires the 'tiny_keccak' feature to be enabled on non-zkvm targets");

#[cfg(not(target_os = "zkvm"))]
mod host_impl;
#[cfg(target_os = "zkvm")]
mod zkvm_impl;

#[cfg(not(target_os = "zkvm"))]
pub use host_impl::{set_keccak256, Keccak256};
#[cfg(target_os = "zkvm")]
pub use zkvm_impl::{native_keccak256, set_keccak256, Keccak256};

#[cfg(feature = "tiny_keccak")]
impl tiny_keccak::Hasher for Keccak256 {
    #[inline(always)]
    fn update(&mut self, input: &[u8]) {
        Keccak256::update(self, input);
    }

    #[inline(always)]
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
