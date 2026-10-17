#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{group::Curve, CurveArithmetic, Group};
use openvm_p256::NistP256;
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_p256::P256Point;

openvm::init!("openvm_init_add.rs");

openvm::entry!(main);

mod test_vectors;
use test_vectors::ADD_TEST_VECTORS;

pub fn main() {
    #[cfg(target_os = "zkvm")]
    {
        use elliptic_curve::subtle::{ConditionallySelectable, ConstantTimeEq};
        use openvm_algebra_guest::IntMod;
        use openvm_p256::{P256Coord, P256Scalar};

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
        check_equality::<P256Coord>();
        check_equality::<P256Scalar>();
        let raw = P256Scalar::from_repr(<P256Scalar as IntMod>::MODULUS);
        let invalid = <P256Scalar as elliptic_curve::PrimeField>::from_repr(
            raw.to_be_bytes_unchecked().into(),
        );
        assert!(bool::from(invalid.is_none()));
        assert_eq!(invalid.unwrap_or(P256Scalar::ONE), P256Scalar::ONE);
    }

    let generator = <NistP256 as CurveArithmetic>::ProjectivePoint::generator();
    let mut p = generator;

    for test_vector in ADD_TEST_VECTORS {
        let affine = p.to_affine();

        let (expected_x, expected_y) = test_vector;
        assert_eq!(&affine.x_be_bytes(), expected_x);
        assert_eq!(&affine.y_be_bytes(), expected_y);

        p += &generator;
    }
}
