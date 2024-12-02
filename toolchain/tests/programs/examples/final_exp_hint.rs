#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use axvm::io::read;
use axvm_ecc::{bls12_381::*, pairing::PairingCheck, AffinePoint};

axvm::entry!(main);

axvm::moduli_init!("0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");

pub fn main() {
    let (p, q, expected): (Vec<AffinePoint<Fp>>, Vec<AffinePoint<Fp2>>, (Fp12, Fp12)) = read();
    let actual = Bls12_381::pairing_check_hint(&p, &q);
    assert_eq!(actual, expected);
}
