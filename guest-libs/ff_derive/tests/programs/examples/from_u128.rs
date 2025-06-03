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

openvm::init!("openvm_init_from_u128.rs");

fn main() {
    use ff::{Field, PrimeField};

    assert_eq!(Bls381K12Scalar::from_u128(1), Bls381K12Scalar::ONE);
    assert_eq!(Bls381K12Scalar::from_u128(2), Bls381K12Scalar::from(2));
    assert_eq!(
        Bls381K12Scalar::from_u128(u128::MAX),
        Bls381K12Scalar::from_str_vartime("340282366920938463463374607431768211455").unwrap(),
    );
}
