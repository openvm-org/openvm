#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
openvm::entry!(main);

use openvm_ff_derive::openvm_prime_field;

extern crate alloc;

/// The BLS12-381 scalar field.
#[openvm_prime_field]
#[PrimeFieldModulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
struct Bls381K12Scalar([u64; 4]);

mod big_endian {
    use openvm_ff_derive::openvm_prime_field;

    #[openvm_prime_field]
    #[PrimeFieldModulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
    #[PrimeFieldGenerator = "7"]
    #[PrimeFieldReprEndianness = "big"]
    struct Bls381K12ScalarBe([u64; 4]);

    pub fn check() {
        super::repr::check::<Bls381K12ScalarBe>(true, 32);
    }
}

#[path = "../repr.rs"]
mod repr;

openvm::init!("openvm_init_from_u128.rs");

// Test arithmetic operations
fn main() {
    repr::check::<Bls381K12Scalar>(false, 32);
    big_endian::check();

    use ff::{Field, PrimeField};

    #[cfg(target_os = "zkvm")]
    {
        use ff::derive::subtle::{ConditionallySelectable, ConstantTimeEq};

        let raw = <Bls381K12Scalar as openvm_algebra_guest::IntMod>::from_le_bytes_unchecked(
            &<Bls381K12Scalar as openvm_algebra_guest::IntMod>::MODULUS,
        );
        assert!(bool::from(raw.ct_eq(&Bls381K12Scalar::ZERO)));
        assert!(bool::from(raw.is_zero()));
        let selected = Bls381K12Scalar::conditional_select(&Bls381K12Scalar::ONE, &raw, 0.into());
        assert_eq!(selected.to_repr(), Bls381K12Scalar::ONE.to_repr());
        let max =
            <Bls381K12Scalar as openvm_algebra_guest::IntMod>::from_le_bytes_unchecked(&[255; 32]);
        assert!(!bool::from(max.ct_eq(&Bls381K12Scalar::ZERO)));
        assert!(!bool::from(Bls381K12Scalar::ZERO.ct_eq(&max)));
        let invalid = ff::derive::subtle::CtOption::new(raw, 0.into());
        assert!(bool::from(invalid.is_none()));
        assert_eq!(
            invalid.unwrap_or(Bls381K12Scalar::ONE),
            Bls381K12Scalar::ONE
        );
    }

    let neg_one = -Bls381K12Scalar::ONE;
    assert_eq!(
        neg_one,
        Bls381K12Scalar::from_str_vartime(
            "52435875175126190479447740508185965837690552500527637822603658699938581184512"
        )
        .unwrap()
    );

    // Test Eq
    #[allow(clippy::eq_op)]
    {
        assert_eq!(Bls381K12Scalar::ZERO, Bls381K12Scalar::ZERO);
        assert_eq!(Bls381K12Scalar::ONE, Bls381K12Scalar::ONE);
        assert_eq!(neg_one, neg_one);
    }

    // Test is_zero
    assert!(bool::from(Bls381K12Scalar::ZERO.is_zero()));
    assert!(!bool::from(Bls381K12Scalar::ONE.is_zero()));

    // Test Add
    assert_eq!(
        neg_one + Bls381K12Scalar::from(10),
        Bls381K12Scalar::from(9)
    );

    // Test AddAssign
    let mut x = neg_one;
    x += Bls381K12Scalar::from(10);
    assert_eq!(x, Bls381K12Scalar::from(9));

    // Test double
    assert_eq!(Bls381K12Scalar::ONE.double(), Bls381K12Scalar::from(2));

    // Test Neg
    assert_eq!(-neg_one, Bls381K12Scalar::from(1));

    // Test Mul
    assert_eq!(
        neg_one * Bls381K12Scalar::from(10),
        -Bls381K12Scalar::from(10)
    );

    // Test MulAssign
    let mut x = neg_one;
    x *= Bls381K12Scalar::from(10);
    assert_eq!(x, -Bls381K12Scalar::from(10));

    // Test Sub
    assert_eq!(
        neg_one - Bls381K12Scalar::from(10),
        -Bls381K12Scalar::from(11)
    );

    // Test SubAssign
    let mut x = neg_one;
    x -= Bls381K12Scalar::from(10);
    assert_eq!(x, -Bls381K12Scalar::from(11));

    // Test Sum
    let sum: Bls381K12Scalar = (0..10).map(Bls381K12Scalar::from).sum();
    assert_eq!(sum, Bls381K12Scalar::from(45));

    // Test Product
    let product: Bls381K12Scalar = (1..10).map(Bls381K12Scalar::from).product();
    assert_eq!(product, Bls381K12Scalar::from(362880));

    // Test Inv
    assert_eq!(
        Bls381K12Scalar::from(2).invert().unwrap(),
        Bls381K12Scalar::TWO_INV
    );
    assert!(bool::from(Bls381K12Scalar::ZERO.invert().is_none()));

    // Test square
    assert_eq!(
        Bls381K12Scalar::TWO_INV.square(),
        Bls381K12Scalar::from(4).invert().unwrap()
    );

    // Test cube
    assert_eq!(
        Bls381K12Scalar::TWO_INV.cube(),
        Bls381K12Scalar::from(8).invert().unwrap()
    );

    // Test Sqrt
    assert!(
        Bls381K12Scalar::from(4).sqrt().unwrap() == Bls381K12Scalar::from(2)
            || Bls381K12Scalar::from(4).sqrt().unwrap() == -Bls381K12Scalar::from(2),
    );
    // by quadratic reciprocity, 5 is not a square since p = 1 mod 12
    assert!(bool::from(Bls381K12Scalar::from(5).sqrt().is_none()));

    // Test pow
    assert_eq!(
        Bls381K12Scalar::from(2).pow([10]),
        Bls381K12Scalar::from(1024)
    );
}
