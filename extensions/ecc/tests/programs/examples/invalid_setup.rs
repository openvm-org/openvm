#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

#[allow(unused_imports)]
use openvm_ecc_guest::{k256::Secp256k1Point, p256::P256Point};

openvm_algebra_moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141",
    "0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff",
    "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"
}

// the order of the curves here does not match the order in supported_curves
openvm_ecc_sw_macros::sw_init! {
    P256Point,
    Secp256k1Point,
}

openvm::entry!(main);

pub fn main() {
    setup_all_moduli();
    // this should cause a debug assertion to fail
    setup_all_sw_curves();
}
