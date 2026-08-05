#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{weierstrass::IntrinsicCurve, CyclicGroup, Group};
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
        let expected = Secp256k1::msm(&[scalar.clone()], &[g.clone()]);

        assert_eq!(via_chip, expected);
        assert_eq!(g.mul_scalar(&scalar), expected);
    }

    // `mul_scalar` accepts an unreduced scalar, which is what callers get from
    // `from_be_bytes_unchecked` on untrusted input. This is `n + 5`, so the product is `5 * G`.
    let unreduced = Secp256k1Scalar::from_be_bytes_unchecked(&hex!(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364146"
    ));
    let expected = Secp256k1::msm(&[Secp256k1Scalar::from_u64(5)], &[g.clone()]);
    assert_eq!(g.mul_scalar(&unreduced), expected);

    // `mul_scalar` also accepts the identity, which the ladder cannot handle for a nonzero scalar.
    let identity = <Secp256k1Point as Group>::IDENTITY;
    assert!(identity
        .mul_scalar(&Secp256k1Scalar::from_u64(7))
        .is_identity());
}
