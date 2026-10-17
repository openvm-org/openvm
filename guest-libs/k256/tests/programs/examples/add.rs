#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{group::Curve, CurveArithmetic, Group};
use openvm_k256::Secp256k1;
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_k256::Secp256k1Point;

mod test_vectors;
use test_vectors::ADD_TEST_VECTORS;

openvm::init!("openvm_init_add.rs");

openvm::entry!(main);

// Taken from https://github.com/RustCrypto/elliptic-curves/blob/32343a78f1522aa5bd856556f114053d4bb938e0/k256/src/arithmetic/projective.rs#L797
pub fn main() {
    #[cfg(target_os = "zkvm")]
    {
        use elliptic_curve::subtle::{ConditionallySelectable, ConstantTimeEq};
        use openvm_algebra_guest::IntMod;
        use openvm_k256::{Secp256k1Coord, Secp256k1Scalar};

        fn check_equality<T: IntMod + ConstantTimeEq + ConditionallySelectable>() {
            let raw = T::from_repr(T::MODULUS);
            assert!(bool::from(raw.ct_eq(&T::ZERO)));
            assert!(bool::from(T::ZERO.ct_eq(&raw)));
            let max = T::from_le_bytes_unchecked(&[255; 32]);
            assert!(!bool::from(max.ct_eq(&T::ZERO)));
            assert!(!bool::from(T::ZERO.ct_eq(&max)));
            let selected = T::conditional_select(&T::ONE, &raw, 0.into());
            assert_eq!(selected.as_le_bytes(), T::ONE.as_le_bytes());
        }
        check_equality::<Secp256k1Coord>();
        check_equality::<Secp256k1Scalar>();
        let raw = Secp256k1Scalar::from_repr(<Secp256k1Scalar as IntMod>::MODULUS);
        let invalid = <Secp256k1Scalar as elliptic_curve::PrimeField>::from_repr(
            raw.to_be_bytes_unchecked().into(),
        );
        assert!(bool::from(invalid.is_none()));
        assert_eq!(
            invalid.unwrap_or(Secp256k1Scalar::ONE),
            Secp256k1Scalar::ONE
        );
    }

    let generator = <Secp256k1 as CurveArithmetic>::ProjectivePoint::generator();
    let mut p = generator;

    for test_vector in ADD_TEST_VECTORS {
        let affine = p.to_affine();

        let (expected_x, expected_y) = test_vector;
        assert_eq!(&affine.x_be_bytes(), expected_x);
        assert_eq!(&affine.y_be_bytes(), expected_y);

        p += &generator;
    }
}
