#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use openvm::io::read;
use openvm_caci::types::{ZkvmKeccak256Hash, ZkvmStatus};

openvm::entry!(main);

extern "C" {
    fn zkvm_keccak256(data: *const u8, len: usize, output: *mut ZkvmKeccak256Hash) -> ZkvmStatus;
}

fn keccak256_c(input: &[u8]) -> [u8; 32] {
    let mut output = ZkvmKeccak256Hash { data: [0u8; 32] };
    let status = unsafe { zkvm_keccak256(input.as_ptr(), input.len(), &mut output) };
    assert!(matches!(status, ZkvmStatus::Ok));
    output.data
}

pub fn main() {
    let num_test_vectors: u32 = read();
    for _ in 0..num_test_vectors {
        let input: Vec<u8> = read();
        let expected_output: Vec<u8> = read();
        assert_eq!(&keccak256_c(&input)[..], &expected_output[..]);
    }

    // NULL data with len == 0 must hash the empty input: `data` is only checked when len > 0.
    let mut output = ZkvmKeccak256Hash { data: [0u8; 32] };
    let status = unsafe { zkvm_keccak256(core::ptr::null(), 0, &mut output) };
    assert!(matches!(status, ZkvmStatus::Ok));
    assert_eq!(output.data, keccak256_c(&[]));

    // NULL data with len > 0 fails without touching `output`.
    let mut output = ZkvmKeccak256Hash { data: [0xaa; 32] };
    let status = unsafe { zkvm_keccak256(core::ptr::null(), 1, &mut output) };
    assert!(matches!(status, ZkvmStatus::Fail));
    assert_eq!(output.data, [0xaa; 32]);

    // NULL output fails.
    let input = [0u8; 1];
    let status = unsafe { zkvm_keccak256(input.as_ptr(), 1, core::ptr::null_mut()) };
    assert!(matches!(status, ZkvmStatus::Fail));
}
