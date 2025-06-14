#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
openvm::entry!(main);

extern crate alloc;

use ff::Field;
use openvm_ff_derive::openvm_prime_field;

// A field modulo a prime such that p = 1 mod 4 and p != 1 mod 16
#[openvm_prime_field]
#[PrimeFieldModulus = "357686312646216567629137"]
#[PrimeFieldGenerator = "5"]
#[PrimeFieldReprEndianness = "little"]
struct Fp([u64; 2]);

fn test(square_root: Fp) {
    let square = square_root.square();
    let square_root = square.sqrt().unwrap();
    assert_eq!(square_root.square(), square);
}

openvm::init!("openvm_init_sqrt.rs");

fn main() {
    test(Fp::ZERO);
    test(Fp::ONE);
    // randomness is not supported in OpenVM
    // use rand::rngs::OsRng;
    // test(Fp::random(OsRng));
    test(Fp::from(1234567890));
}
