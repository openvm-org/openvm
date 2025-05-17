#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use openvm::io::read;
use openvm_ecc_guest::AffinePoint;
use openvm_pairing::{
    bls12_381::{Bls12_381, Fp, Fp12, Fp2},
    PairingCheck,
};

openvm::entry!(main);

openvm::init!("openvm_init_bls_final_exp_hint_bls12_381.rs");

pub fn main() {
    #[allow(clippy::type_complexity)]
    let (p, q, expected): (Vec<AffinePoint<Fp>>, Vec<AffinePoint<Fp2>>, (Fp12, Fp12)) = read();
    let actual = Bls12_381::pairing_check_hint(&p, &q);
    assert_eq!(actual, expected);
}
