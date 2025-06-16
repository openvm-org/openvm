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

openvm::init!("openvm_init_constants.rs");

fn main() {
    use ff::{Field, PrimeField};

    assert_eq!(
        Bls381K12Scalar::MODULUS,
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    );

    assert_eq!(
        Bls381K12Scalar::from(2) * Bls381K12Scalar::TWO_INV,
        Bls381K12Scalar::ONE,
    );

    assert_eq!(
        Bls381K12Scalar::ROOT_OF_UNITY * Bls381K12Scalar::ROOT_OF_UNITY_INV,
        Bls381K12Scalar::ONE,
    );

    // ROOT_OF_UNITY^{2^s} mod m == 1
    assert_eq!(
        Bls381K12Scalar::ROOT_OF_UNITY.pow([1u64 << Bls381K12Scalar::S, 0, 0, 0]),
        Bls381K12Scalar::ONE,
    );

    // DELTA^{t} mod m == 1
    assert_eq!(
        Bls381K12Scalar::DELTA.pow([
            0xfffe5bfeffffffff,
            0x09a1d80553bda402,
            0x299d7d483339d808,
            0x73eda753,
        ]),
        Bls381K12Scalar::ONE,
    );
}
