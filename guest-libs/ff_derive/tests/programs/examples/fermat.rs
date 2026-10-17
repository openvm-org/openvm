#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

extern crate alloc;

use openvm_ff_derive::openvm_prime_field;

/// The largest known Fermat prime, used to test the case `t = 1`.
#[openvm_prime_field]
#[PrimeFieldModulus = "65537"]
#[PrimeFieldGenerator = "3"]
#[PrimeFieldReprEndianness = "little"]
struct Fermat65537Field([u64; 1]);

mod big_endian {
    use openvm_ff_derive::openvm_prime_field;

    #[openvm_prime_field]
    #[PrimeFieldModulus = "65537"]
    #[PrimeFieldGenerator = "3"]
    #[PrimeFieldReprEndianness = "big"]
    struct Fermat65537FieldBe([u64; 1]);

    pub fn check() {
        super::repr::check::<Fermat65537FieldBe>(true, 32);
    }
}

#[path = "../repr.rs"]
mod repr;

openvm::init!("openvm_init_fermat.rs");

fn main() {
    repr::check::<Fermat65537Field>(false, 32);
    big_endian::check();
}
