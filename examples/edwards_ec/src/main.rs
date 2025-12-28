// [!region imports]
use hex_literal::hex;
use openvm as _;
use openvm_algebra_guest::IntMod;
use openvm_te_guest::{
    ed25519::{Ed25519Coord, Ed25519Point},
    edwards::TwistedEdwardsPoint,
};
// [!endregion imports]

// [!region init]
openvm::init!();
/* The init! macro will expand to the following
openvm_algebra_guest::moduli_macros::moduli_init! {
    "0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED",
    "0x1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED"
}
openvm_te_guest::te_macros::te_init! { "Ed25519Point" }
*/
// [!endregion init]

// [!region main]
pub fn main() {
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
