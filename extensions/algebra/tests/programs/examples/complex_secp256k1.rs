#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm_algebra_guest::{field::ComplexConjugate, DivAssignUnsafe, DivUnsafe, IntMod};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Secp256k1Coord { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F" }
}

openvm_algebra_complex_macros::complex_declare! {
    Complex { mod_type = Secp256k1Coord }
}

openvm::init!("openvm_init_complex_secp256k1.rs");

pub fn main() {
    let mut a = Complex::new(
        Secp256k1Coord::from_repr(core::array::from_fn(|_| 10)),
        Secp256k1Coord::from_repr(core::array::from_fn(|_| 21)),
    );
    let mut b = Complex::new(
        Secp256k1Coord::from_repr(core::array::from_fn(|_| 32)),
        Secp256k1Coord::from_repr(core::array::from_fn(|_| 47)),
    );

    for _ in 0..32 {
        let mut res = &a * &b;
        res += &a * &Complex::new(Secp256k1Coord::ZERO, -b.c1.double());
        res.div_assign_unsafe(&b * &b.clone().conjugate());

        if a.clone().div_unsafe(&b) - res != Complex::ZERO {
            panic!();
        }

        a *= &b;
        b *= &a;
    }

    if a == b {
        panic!();
    }
}
