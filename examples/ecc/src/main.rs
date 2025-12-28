// [!region imports]
use hex_literal::hex;
use openvm as _;
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::weierstrass::WeierstrassPoint;
use openvm_k256::{Secp256k1Coord, Secp256k1Point};
use openvm_te_guest::{
    ed25519::{Ed25519Coord, Ed25519Point},
    edwards::TwistedEdwardsPoint,
};
// [!endregion imports]

// [!region init]
openvm::init!();
/* The init! macro will expand to the following
openvm_algebra_guest::moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
}
openvm_ecc_guest::sw_macros::sw_init! { "Secp256k1Point" }
openvm_ecc_guest::te_macros::te_init! { "Ed25519Point" }
*/
// [!endregion init]

// [!region main]
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

    let x1 = Ed25519Coord::from_be_bytes_unchecked(&hex!(
        "216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A"
    ));
    let y1 = Ed25519Coord::from_be_bytes_unchecked(&hex!(
        "6666666666666666666666666666666666666666666666666666666666666658"
    ));
    let p1 = Ed25519Point::from_xy(x1, y1).unwrap();

    let x2 = Ed25519Coord::from_u32(2);
    let y2 = Ed25519Coord::from_be_bytes_unchecked(&hex!(
        "1A43BF127BDDC4D71FF910403C11DDB5BA2BCDD2815393924657EF111E712631"
    ));
    let p2 = Ed25519Point::from_xy(x2, y2).unwrap();

    #[allow(clippy::op_ref)]
    let _p3 = &p1 + &p2;
}
// [!endregion main]
