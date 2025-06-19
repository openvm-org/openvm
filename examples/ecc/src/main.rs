// ANCHOR: imports
use hex_literal::hex;
use openvm_ecc_guest::{
    algebra::IntMod,
    ed25519::{Ed25519Coord, Ed25519Point},
    edwards::TwistedEdwardsPoint,
    weierstrass::WeierstrassPoint,
};
use openvm_k256::{Secp256k1Coord, Secp256k1Point};
// ANCHOR_END: imports

// ANCHOR: init
openvm::init!();
/* The init! macro will expand to the following
openvm_algebra_guest::moduli_macros::moduli_init! {
"115792089237316195423570985008687907853269984665640564039457584007908834671663",
"115792089237316195423570985008687907852837564279074904382605163141518161494337"
}
openvm_ecc_guest::sw_macros::sw_init! { Secp256k1Point }
openvm_ecc_guest::te_macros::te_init! { Ed25519Point }
*/
// ANCHOR_END: init

// ANCHOR: main
pub fn main() {
    let x1 = Secp256k1Coord::from_u32(1);
    let y1 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "EEA7767E580D75BC6FDD7F58D2A84C2614FB22586068DB63B346C6E60AF21842"
    ));
    let p1 = Secp256k1Point::from_xy_nonidentity(x1, y1).unwrap();

    let x2 = Secp256k1Coord::from_u32(2);
    let y2 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "D1A847A8F879E0AEE32544DA5BA0B3BD1703A1F52867A5601FF6454DD8180499"
    ));
    let p2 = Secp256k1Point::from_xy_nonidentity(x2, y2).unwrap();

    #[allow(clippy::op_ref)]
    let _p3 = &p1 + &p2;

    let x1 = Ed25519Coord::from_be_bytes(&hex!(
        "216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A"
    ));
    let y1 = Ed25519Coord::from_be_bytes(&hex!(
        "6666666666666666666666666666666666666666666666666666666666666658"
    ));
    let p1 = Ed25519Point::from_xy(x1, y1).unwrap();

    let x2 = Ed25519Coord::from_u32(2);
    let y2 = Ed25519Coord::from_be_bytes(&hex!(
        "1A43BF127BDDC4D71FF910403C11DDB5BA2BCDD2815393924657EF111E712631"
    ));
    let p2 = Ed25519Point::from_xy(x2, y2).unwrap();

    #[allow(clippy::op_ref)]
    let _p3 = &p1 + &p2;
}
// ANCHOR_END: main
