#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{weierstrass::CachedMulTable, CyclicGroup, Group};
use openvm_k256::{Secp256k1, Secp256k1Point, Secp256k1Scalar};

openvm::init!("openvm_init_ec_mul_k256.rs");

openvm::entry!(main);

fn windowed_reference(base: Secp256k1Point, scalar: Secp256k1Scalar) -> Secp256k1Point {
    let base = [base];
    let table = CachedMulTable::<Secp256k1>::new_with_prime_order(&base, 4);
    table.windowed_mul(&[scalar])
}

pub fn main() {
    let g = Secp256k1Point::GENERATOR;
    let neg_g = Secp256k1Point::NEG_GENERATOR;
    let generic = g.double();

    // The intrinsic requires odd scalars below the group order.
    for k in [1u64, 3, 5, 255, 0x1234_5679, u32::MAX as u64] {
        let scalar = Secp256k1Scalar::from_u64(k);
        let bytes: [u8; 32] = scalar.as_le_bytes().try_into().unwrap();

        for base in [g.clone(), neg_g.clone(), generic.clone()] {
            let via_chip = unsafe { base.mul_scalar_le_unchecked::<true>(&bytes) };
            assert_eq!(via_chip, windowed_reference(base, scalar.clone()));
        }
    }

    // `mul_scalar` is total: zero and even scalars route through the n - k substitution.
    for k in [0u64, 1, 2, 3, 255, 0x1234_5678, u32::MAX as u64] {
        let scalar = Secp256k1Scalar::from_u64(k);
        assert_eq!(g.mul_scalar(&scalar), windowed_reference(g.clone(), scalar));
    }

    // n + 5, an unreduced scalar
    let unreduced = Secp256k1Scalar::from_be_bytes_unchecked(&hex!(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364146"
    ));
    let expected = windowed_reference(g.clone(), Secp256k1Scalar::from_u64(5));
    assert_eq!(g.mul_scalar(&unreduced), expected);

    let identity = <Secp256k1Point as Group>::IDENTITY;
    assert!(identity
        .mul_scalar(&Secp256k1Scalar::from_u64(7))
        .is_identity());
}
