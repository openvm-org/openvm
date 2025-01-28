#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::read_vec;
use openvm_ecc_guest::{
    algebra::IntMod,
    ed25519::{Ed25519Coord, Ed25519Point},
    edwards::TwistedEdwardsPoint,
    k256::{Secp256k1Coord, Secp256k1Point},
    weierstrass::WeierstrassPoint,
    FromCompressed,
};

openvm_algebra_moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141",
    "57896044618658097711785492504343953926634992332820282019728792003956564819949",
}
openvm_ecc_sw_macros::sw_init! {
    Secp256k1Point,
}

openvm_ecc_te_setup::te_init! {
    Ed25519Point,
}

openvm::entry!(main);

pub fn main() {
    setup_0();
    setup_2();
    setup_all_sw_curves();
    setup_all_te_curves();

    let bytes = read_vec();

    // secp256k1
    let x = Secp256k1Coord::from_le_bytes(&bytes[..32]);
    let y = Secp256k1Coord::from_le_bytes(&bytes[32..64]);
    let rec_id = y.as_le_bytes()[0] & 1;

    let hint_y = Secp256k1Point::hint_decompress(&x, &rec_id);
    assert_eq!(y, hint_y);

    let p = Secp256k1Point::decompress(x.clone(), &rec_id);
    assert_eq!(p.x(), &x);
    assert_eq!(p.y(), &y);

    // ed25519
    let x = Ed25519Coord::from_le_bytes(&bytes[64..96]);
    let y = Ed25519Coord::from_le_bytes(&bytes[96..128]);
    let rec_id = x.as_le_bytes()[0] & 1;

    let hint_x = Ed25519Point::hint_decompress(&y, &rec_id);
    assert_eq!(x, hint_x);

    let p = Ed25519Point::decompress(y.clone(), &rec_id);
    assert_eq!(p.x(), &x);
    assert_eq!(p.y(), &y);
}
