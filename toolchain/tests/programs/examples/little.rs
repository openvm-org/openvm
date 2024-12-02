#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm_algebra_guest::{DivUnsafe, IntMod};
use axvm_ecc_guest::sw::Secp256k1Coord;

axvm::entry!(main);

pub fn main() {
    let mut pow = Secp256k1Coord::MODULUS;
    pow[0] -= 2;

    let mut a = Secp256k1Coord::from_u32(1234);
    let mut res = Secp256k1Coord::from_u32(1);
    let inv = res.clone().div_unsafe(&a);

    for i in 0..32 {
        for j in 0..8 {
            if pow[i] & (1 << j) != 0 {
                res = res * &a;
            }
            a *= a.clone();
        }
    }

    // https://en.wikipedia.org/wiki/Fermat%27s_little_theorem
    if res != inv {
        axvm::process::panic();
    }

    let two = Secp256k1Coord::from_u32(2);
    let minus_two = Secp256k1Coord::from_le_bytes(&pow);

    if (res - &minus_two) != (inv + &two) {
        axvm::process::panic();
    }

    if two == minus_two {
        axvm::process::panic();
    }
}
