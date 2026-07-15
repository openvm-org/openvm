#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm_algebra_guest::{DivUnsafe, IntMod};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Bls12381Fq { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" }
}

openvm::init!("openvm_init_rvr_modular_48.rs");

pub fn main() {
    let mut a = Bls12381Fq::from_u32(7);
    let b = Bls12381Fq::from_u32(9);

    for _ in 0..20 {
        let sum = a.clone() + &b;
        let difference = sum.clone() - &b;
        assert_eq!(difference, a);
        let product = difference * &b;
        let quotient = product.div_unsafe(&b);
        assert_eq!(quotient, a);
        a = sum;
    }
}
