#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm_algebra_guest::{DivUnsafe, IntMod};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Secp256k1Coord { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F" }
}

openvm::init!("openvm_init_rvr_modular.rs");

pub fn main() {
    let mut a = Secp256k1Coord::from_u32(7);
    let b = Secp256k1Coord::from_u32(9);

    for _ in 0..100 {
        let sum = a.clone() + &b;
        let difference = sum.clone() - &b;
        assert_eq!(difference, a);
        let product = difference * &b;
        let quotient = product.div_unsafe(&b);
        assert_eq!(quotient, a);
        a = sum;
    }
}
