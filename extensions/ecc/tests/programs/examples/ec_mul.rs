#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{weierstrass::IntrinsicCurve, CyclicGroup};
use openvm_k256::{Secp256k1, Secp256k1Point, Secp256k1Scalar};

openvm::init!("openvm_init_ec_mul_k256.rs");

openvm::entry!(main);

pub fn main() {
    let g = Secp256k1Point::GENERATOR;

    // Scalars are chosen to reach each of the ladder's cases: an all-zero scalar stays at the
    // identity throughout, small values exercise the transition out of it, and a full-width value
    // interleaves doublings with additions. All are well below the group order, which
    // `mul_scalar_le_unchecked` requires.
    let scalars = [0u64, 1, 2, 3, 255, 0x1234_5678, u32::MAX as u64];

    for k in scalars {
        let scalar = Secp256k1Scalar::from_u64(k);
        let bytes: [u8; 32] = scalar.as_le_bytes().try_into().unwrap();

        let via_chip = unsafe { g.mul_scalar_le_unchecked(&bytes) };
        let expected = Secp256k1::msm(&[scalar], &[g.clone()]);

        assert_eq!(via_chip, expected);
    }
}
