#![no_std]

#[cfg(any(openvm_intrinsics, target_os = "openvm", feature = "tiny_keccak"))]
use openvm_keccak256_guest::KECCAK_OUTPUT_SIZE;

#[cfg(all(
    not(any(openvm_intrinsics, target_os = "openvm")),
    feature = "tiny_keccak"
))]
mod host_impl;
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
mod zkvm_impl;

#[cfg(all(
    not(any(openvm_intrinsics, target_os = "openvm")),
    feature = "tiny_keccak"
))]
pub use host_impl::{set_keccak256, Keccak256};
#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
pub use zkvm_impl::{native_keccak256, set_keccak256, Keccak256};

#[cfg(all(
    not(any(openvm_intrinsics, target_os = "openvm")),
    feature = "tiny_keccak"
))]
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
#[cfg(any(openvm_intrinsics, target_os = "openvm", feature = "tiny_keccak"))]
#[inline(always)]
pub fn keccak256(input: &[u8]) -> [u8; KECCAK_OUTPUT_SIZE] {
    let mut output = [0u8; KECCAK_OUTPUT_SIZE];
    set_keccak256(input, &mut output);
    output
}
