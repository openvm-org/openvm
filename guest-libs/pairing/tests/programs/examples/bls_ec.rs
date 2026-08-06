#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::reveal_bytes32;
use openvm_ecc_guest::{algebra::IntMod, weierstrass::WeierstrassPoint, CyclicGroup, Group};
use openvm_pairing::bls12_381::Bls12_381G1Affine;

openvm::init!("openvm_init_bls_ec_bls12_381.rs");

openvm::entry!(main);

const EXPECTED_DIGEST: [u8; 32] = [
    0x03, 0x54, 0x29, 0x16, 0x48, 0xb6, 0x17, 0xa6, 0x10, 0x6a, 0x3e, 0x24, 0x66, 0xc1, 0x91, 0xaf,
    0x4e, 0x1d, 0x55, 0x30, 0x37, 0x24, 0x7e, 0xd0, 0xe8, 0xaf, 0x94, 0xbd, 0x58, 0x9b, 0xfb, 0x8c,
];

fn absorb(digest: &mut [u8; 32], position: &mut usize, bytes: &[u8]) {
    for &byte in bytes {
        let slot = *position % digest.len();
        digest[slot] = digest[slot].rotate_left(1) ^ byte;
        *position += 1;
    }
}

pub fn main() {
    let doubled = Bls12_381G1Affine::GENERATOR.double();
    let sum = &doubled + &Bls12_381G1Affine::GENERATOR;

    let mut digest = [0u8; 32];
    let mut position = 0;
    for result in [&doubled, &sum] {
        // Hash the affine coordinates (normalize to z = 1) so the digest reflects the point,
        // not the incidental projective (X, Y, Z) encoding.
        let affine = result.normalize();
        absorb(&mut digest, &mut position, affine.x().as_le_bytes());
        absorb(&mut digest, &mut position, affine.y().as_le_bytes());
    }
    assert_eq!(digest, EXPECTED_DIGEST);
    reveal_bytes32(digest);
}
