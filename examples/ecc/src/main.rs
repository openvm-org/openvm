// ANCHOR: imports
use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{
    k256::{Secp256k1Coord, Secp256k1Point},
    weierstrass::WeierstrassPoint,
};
// ANCHOR_END: imports

// ANCHOR: init
openvm_algebra_guest::moduli_macros::moduli_init! {
    "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
    "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"
}

openvm_ecc_guest::sw_macros::sw_init! {
    Secp256k1Point,
}
// ANCHOR_END: init

// ANCHOR: main
pub fn main() {
    setup_all_moduli();
    setup_all_curves();

    // Generator point G (valid on secp256k1)
    let x1 = Secp256k1Coord::from_be_bytes(&hex!(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    ));
    let y1 = Secp256k1Coord::from_be_bytes(&hex!(
        "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
    ));
    let p1 = Secp256k1Point::generator();

    // 2G (also valid)
    let p2 = &p1 + &p1;

    // Use the points
    let _p3 = &p1 + &p2;
}
// ANCHOR_END: main
