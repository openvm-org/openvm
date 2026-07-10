#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use openvm_ecc_guest::{CyclicGroup, Group};
use openvm_pairing::bls12_381::Bls12_381G1Affine;

openvm::init!("openvm_init_ec_bls12_381_bls12_381.rs");

openvm::entry!(main);

pub fn main() {
    let generator = Bls12_381G1Affine::GENERATOR;
    black_box(generator.clone() + generator.clone());
    black_box(generator.double());
}
