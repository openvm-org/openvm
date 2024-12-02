#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use axvm::io::read;
use axvm_ecc_guest::{sw::setup_fp2, AffinePoint};
use axvm_pairing_guest::{bls12_381::*, pairing::PairingCheck};

axvm::entry!(main);

pub fn main() {
    setup_Bls12_381Fp();
    setup_Bls12_381Fp_fp2();

    let (p, q, expected): (Vec<AffinePoint<Fp>>, Vec<AffinePoint<Fp2>>, (Fp12, Fp12)) = read();
    let actual = Bls12_381::pairing_check_hint(&p, &q);
    assert_eq!(actual, expected);
}
