#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

#[allow(unused_imports)]
use openvm_pairing_guest::bls12_381::Bls12_381G1Affine;

openvm_algebra_moduli_macros::moduli_init! {
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
    "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
}

openvm_ecc_sw_macros::sw_init! {
    Bls12_381G1Affine,
}

openvm::entry!(main);

pub fn main() {
    setup_all_moduli();
    setup_all_curves();
}
