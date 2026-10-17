#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
openvm::entry!(main);

extern crate alloc;

use ff::{Field, PrimeField};
use openvm_ff_derive::openvm_prime_field;

#[openvm_prime_field]
#[PrimeFieldModulus = "39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319"]
#[PrimeFieldGenerator = "19"]
#[PrimeFieldReprEndianness = "little"]
struct F384p([u64; 7]);

mod big_endian {
    use openvm_ff_derive::openvm_prime_field;

    #[openvm_prime_field]
    #[PrimeFieldModulus = "39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319"]
    #[PrimeFieldGenerator = "19"]
    #[PrimeFieldReprEndianness = "big"]
    struct F384pBe([u64; 7]);

    pub fn check() {
        super::repr::check::<F384pBe>(true, 48);
    }
}

#[path = "../repr.rs"]
mod repr;

fn test(square_root: F384p) {
    let square = square_root.square();
    let square_root = square.sqrt().unwrap();
    assert_eq!(square_root.square(), square);
}

openvm::init!("openvm_init_full_limbs.rs");

// Test that random masking does not overflow
fn main() {
    repr::check::<F384p>(false, 48);
    big_endian::check();

    use ff::Field;

    // randomness is not supported in OpenVM
    // use rand::rngs::OsRng;
    // let _ = F384p::random(OsRng);

    test(F384p::ZERO);
    test(F384p::ONE);
    test(F384p::from(1234567890));
    test(F384p::from_str_vartime("19402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319").unwrap());
}
