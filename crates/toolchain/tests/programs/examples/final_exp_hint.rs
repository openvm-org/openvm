#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use axvm::io::read;
use axvm_ecc_guest::AffinePoint;
use axvm_pairing_guest::{
    bls12_381::{Bls12_381, Fp, Fp12, Fp2, Scalar},
    pairing::PairingCheck,
};

axvm::entry!(main);

axvm_algebra_moduli_setup::moduli_init! {
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
    "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
}

pub fn main() {
    setup_all_moduli();
    let (p, q, expected): (Vec<AffinePoint<Fp>>, Vec<AffinePoint<Fp2>>, (Fp12, Fp12)) = read();
    let actual = Bls12_381::pairing_check_hint(&p, &q);
    assert_eq!(actual, expected);
}
