#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

#[allow(unused_imports)]
#[cfg(feature = "bn254")]
use openvm_pairing::bn254::Bn254G1Affine;

openvm::init!("openvm_init_bn_ec_bn254.rs");

openvm::entry!(main);

pub fn main() {}
