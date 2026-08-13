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

/// Reference implementation: the windowed table, called directly.
fn windowed_reference(base: Secp256k1Point, scalar: Secp256k1Scalar) -> Secp256k1Point {
    let base = [base];
    let table = CachedMulTable::<Secp256k1>::new_with_prime_order(&base, 4);
    table.windowed_mul(&[scalar])
}

pub fn main() {
    let g = Secp256k1Point::GENERATOR;
    let neg_g = Secp256k1Point::NEG_GENERATOR;
    let generic = g.double();

    // The intrinsic requires an odd scalar below the group order. Its digits are all `+-1`, whose
    // sum is odd for any choice of signs, so an even scalar has no digit assignment at all. These
    // reach the small multipliers, where the accumulator is `+-P` and the step ordering matters,
    // and a full-width one that interleaves both signs.
    for k in [1u64, 3, 5, 255, 0x1234_5679, u32::MAX as u64] {
        let scalar = Secp256k1Scalar::from_u64(k);
        let bytes: [u8; 32] = scalar.as_le_bytes().try_into().unwrap();

        for base in [g.clone(), neg_g.clone(), generic.clone()] {
            let via_chip = unsafe { base.mul_scalar_le_unchecked(&bytes) };
            assert_eq!(via_chip, windowed_reference(base, scalar.clone()));
        }
    }

    // `mul_scalar` discharges that precondition, so it is total. Zero and the even scalars below
    // route through the `n - k` substitution and a negation.
    for k in [0u64, 1, 2, 3, 255, 0x1234_5678, u32::MAX as u64] {
        let scalar = Secp256k1Scalar::from_u64(k);
        assert_eq!(g.mul_scalar(&scalar), windowed_reference(g.clone(), scalar));
    }

    // `mul_scalar` accepts an unreduced scalar, which is what callers get from
    // `from_be_bytes_unchecked` on untrusted input. This is `n + 5`, so the product is `5 * G`.
    let unreduced = Secp256k1Scalar::from_be_bytes_unchecked(&hex!(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364146"
    ));
    let expected = windowed_reference(g.clone(), Secp256k1Scalar::from_u64(5));
    assert_eq!(g.mul_scalar(&unreduced), expected);

    // `mul_scalar` also accepts the identity, which the ladder cannot handle for a nonzero scalar.
    let identity = <Secp256k1Point as Group>::IDENTITY;
    assert!(identity
        .mul_scalar(&Secp256k1Scalar::from_u64(7))
        .is_identity());
}
