#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use openvm::io::read;
use openvm_ecc_guest::AffinePoint;
use openvm_pairing_guest::{
    bn254::{Bn254, Fp, Fp12, Fp2},
    pairing::PairingCheck,
};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_init! {
    "21888242871839275222246405745257275088696311157297823662689037894645226208583",
    "21888242871839275222246405745257275088548364400416034343698204186575808495617",
}

pub fn main() {
    #[allow(clippy::type_complexity)]
    let (p, q, expected): (Vec<AffinePoint<Fp>>, Vec<AffinePoint<Fp2>>, (Fp12, Fp12)) = read();
    let actual = Bn254::pairing_check_hint(&p, &q);
    assert_eq!(actual, expected);
}
