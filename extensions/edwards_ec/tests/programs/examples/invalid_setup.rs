#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;
use openvm_te_guest::{
    ed25519::{Ed25519Coord, Ed25519Point},
    CyclicGroup,
};

pub const CURVE_A: Ed25519Coord = Ed25519Coord::from_const_bytes(le_bytes_from_const_u8(3));
pub const CURVE_D: Ed25519Coord = Ed25519Coord::from_const_bytes(le_bytes_from_const_u8(2));
// Workaround for the fact that Ed25519Coord::from_const_u8 is private
pub const fn le_bytes_from_const_u8(value: u8) -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[0] = value;
    buf
}

openvm_te_guest::te_macros::te_declare! {
    SampleCurvePoint { mod_type = Ed25519Coord, a = CURVE_A, d = CURVE_D },
}

openvm_algebra_moduli_macros::moduli_init! {
    "0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED", // ED25519_MODULUS
    "0x1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED", // ED25519_ORDER
    "3", // CURVE_A
    "1", // order is unknown, set to 1 (this doesn't really matter since we aren't using pairing extension)
}

// the order of the curves here does not match the order in supported_curves
openvm_te_guest::te_macros::te_init! {
    "SampleCurvePoint",
    "Ed25519Point",
}

openvm::entry!(main);

pub fn main() {
    // this should cause a debug assertion to fail
    let p1 = Ed25519Point::GENERATOR;
    let _p2 = &p1 + &p1;
}
