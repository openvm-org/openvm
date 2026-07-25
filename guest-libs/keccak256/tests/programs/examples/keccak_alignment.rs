#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use openvm_keccak256::{keccak256, Keccak256};

openvm::entry!(main);

/// Input lengths covering every residue modulo the XORIN instruction's 8-byte word, both
/// sides of the 136-byte rate boundary, and multi-block inputs.
const LENGTHS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 20, 31, 32, 33, 63, 64, 65, 127, 128, 134, 135, 136,
    137, 138, 143, 144, 145, 271, 272, 273, MAX_LEN,
];

const MAX_LEN: usize = 400;

/// Chunk sizes for incremental absorbs. Sizes that are not multiples of 8 walk the sponge's
/// fill index off the word boundary and keep it there for the rest of the hash.
const CHUNKS: &[usize] = &[1, 3, 5, 7, 8, 9, 33, 135, 136, 137];

const EMPTY_DIGEST: [u8; 32] = [
    0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
    0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b, 0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
];

const ABC_DIGEST: [u8; 32] = [
    0x4e, 0x03, 0x65, 0x7a, 0xea, 0x45, 0xa9, 0x4f, 0xc7, 0xd4, 0x7b, 0xa8, 0x26, 0xc8, 0xd6, 0x67,
    0xc0, 0xd1, 0xe6, 0xe3, 0x3a, 0x64, 0xa0, 0x36, 0xec, 0x44, 0xf5, 0x8f, 0xa1, 0x2d, 0x6c, 0x45,
];

/// A buffer whose first byte is 8-byte aligned, so that indexing it by a byte offset yields a
/// slice with exactly that misalignment.
#[repr(align(8))]
struct AlignedBuf([u8; MAX_LEN + 8]);

fn sample_bytes(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
        .collect()
}

/// Hashes `data` placed `offset` bytes past an 8-byte boundary.
fn digest_at_offset(data: &[u8], offset: usize) -> [u8; 32] {
    let mut buf = AlignedBuf([0u8; MAX_LEN + 8]);
    assert_eq!(
        buf.0.as_ptr() as usize % 8,
        0,
        "test buffer is not 8-byte aligned"
    );
    let region = &mut buf.0[offset..offset + data.len()];
    region.copy_from_slice(data);
    keccak256(black_box(region))
}

pub fn main() {
    // Known answers: a uniformly wrong implementation would still satisfy the
    // self-consistency checks below.
    assert_eq!(keccak256(black_box(b"")), EMPTY_DIGEST);
    assert_eq!(keccak256(black_box(b"abc")), ABC_DIGEST);

    for &len in LENGTHS {
        let data = sample_bytes(len);

        // The digest must not depend on where the input sits relative to the word boundary.
        let expected = digest_at_offset(&data, 0);
        for offset in 1..8 {
            assert_eq!(
                digest_at_offset(&data, offset),
                expected,
                "digest changed with input alignment"
            );
        }

        // Absorbing in chunks must agree with absorbing all at once, including when the
        // chunk boundaries leave the sponge's fill index unaligned.
        for &chunk in CHUNKS {
            let mut hasher = Keccak256::new();
            for piece in data.chunks(chunk) {
                hasher.update(black_box(piece));
            }
            let mut output = [0u8; 32];
            hasher.finalize(&mut output);
            assert_eq!(output, expected, "incremental digest disagreed");
        }
    }
}
