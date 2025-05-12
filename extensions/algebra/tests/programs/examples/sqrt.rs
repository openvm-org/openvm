#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm_algebra_guest::{Field, IntMod, Sqrt};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Secp256k1Coord {
        modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    }
}

openvm::init!("openvm_init_sqrt.rs");

pub fn main() {
    let a = Secp256k1Coord::from_u32(4);
    let sqrt = a.sqrt();
    assert_eq!(sqrt, Some(Secp256k1Coord::from_u32(2)));

    let b = <Secp256k1Coord as IntMod>::ZERO - <Secp256k1Coord as IntMod>::ONE;
    let sqrt = b.sqrt();
    // -1 is not a quadratic residue modulo p when p = 3 mod 4
    // See https://math.stackexchange.com/questions/735400/if-p-equiv-3-mod-4-with-p-prime-prove-1-is-a-non-quadratic-residue-modulo
    assert_eq!(sqrt, None);

    let expected = b * Secp256k1Coord::from_u32(2).invert();
    let c = expected.square();
    let result = c.sqrt();
    assert!(result == Some(expected.clone()) || result == Some(-expected));
}
