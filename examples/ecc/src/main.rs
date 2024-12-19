#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_algebra_guest::IntMod;

openvm::entry!(main);

use hex_literal::hex;
use openvm_ecc_guest::{
    k256::{Secp256k1Coord, Secp256k1Point},
    weierstrass::WeierstrassPoint,
};

openvm_algebra_guest::moduli_setup::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}

openvm_ecc_guest::sw_setup::sw_init! {
    Secp256k1Coord,
}

pub fn main() {
    setup_all_moduli();
    setup_all_curves();
    let x1 = Secp256k1Coord::from_u32(1);
    let y1 = Secp256k1Coord::from_le_bytes(&hex!(
        "EEA7767E580D75BC6FDD7F58D2A84C2614FB22586068DB63B346C6E60AF21842"
    ));
    let p1 = Secp256k1Point::from_xy_nonidentity(x1, y1).unwrap();

    let x2 = Secp256k1Coord::from_u32(2);
    let y2 = Secp256k1Coord::from_le_bytes(&hex!(
        "D1A847A8F879E0AEE32544DA5BA0B3BD1703A1F52867A5601FF6454DD8180499"
    ));
    let p2 = Secp256k1Point::from_xy_nonidentity(x2, y2).unwrap();

    let _p3 = &p1 + &p2;
}
