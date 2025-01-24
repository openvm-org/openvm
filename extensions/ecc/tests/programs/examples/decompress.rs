#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::read_vec;
use openvm_ecc_guest::{
    algebra::IntMod,
    k256::{Secp256k1Coord, Secp256k1Point},
    weierstrass::{FromCompressed, WeierstrassPoint},
};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}
openvm_ecc_sw_macros::sw_init! {
    Secp256k1Point,
}

pub fn main() {
    setup_0();
    setup_all_curves();

    let bytes = read_vec();
    let x = Secp256k1Coord::from_le_bytes(&bytes[..32]);
    let y = Secp256k1Coord::from_le_bytes(&bytes[32..]);
    let rec_id = y.as_le_bytes()[0] & 1;

    let hint_y = Secp256k1Point::hint_decompress(&x, &rec_id);
    assert_eq!(y, hint_y);

    let p = Secp256k1Point::decompress(x.clone(), &rec_id);
    assert_eq!(p.x(), &x);
    assert_eq!(p.y(), &y);
}
