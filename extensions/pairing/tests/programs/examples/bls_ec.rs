#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

#[allow(unused_imports)]
use openvm_pairing_guest::bls12_381::Bls12_381G1Affine;

openvm::init!("openvm_init_bls_ec_bls12_381.rs");

openvm::entry!(main);

pub fn main() {
    setup_all_moduli();
    setup_all_curves();
}
