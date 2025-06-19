#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use openvm_algebra_guest::{DivUnsafe, IntMod};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    Secp256k1Coord { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F" }
}

openvm::init!("openvm_init_little.rs");

pub fn main() {
    let mut pow = Secp256k1Coord::MODULUS;
    pow[0] -= 2;

    let mut a = Secp256k1Coord::from_u32(1234);
    let mut res = Secp256k1Coord::from_u32(1);
    let inv = res.clone().div_unsafe(&a);

    for pow_bit in pow {
        for j in 0..8 {
            if pow_bit & (1 << j) != 0 {
                res *= &a;
            }
            a *= a.clone();
        }
    }

    // https://en.wikipedia.org/wiki/Fermat%27s_little_theorem
    assert_eq!(res, inv);

    let two = Secp256k1Coord::from_u32(2);
    let minus_two = Secp256k1Coord::from_le_bytes_unchecked(&pow);

    assert_eq!(res - &minus_two, inv + &two);

    if two == minus_two {
        openvm::process::panic();
    }
}
