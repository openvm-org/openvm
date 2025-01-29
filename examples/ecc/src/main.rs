// ANCHOR: imports
use hex_literal::hex;
use openvm_algebra_guest::{Field, IntMod};
use openvm_ecc_guest::{
    edwards::TwistedEdwardsPoint,
    weierstrass::WeierstrassPoint
    Group,
};
use openvm_k256::{Secp256k1Coord, Secp256k1Point};
// ANCHOR_END: imports
openvm_algebra_guest::moduli_macros::moduli_declare! {
    // The Secp256k1 modulus and scalar field modulus are already declared in the k256 module
    Edwards25519Coord { modulus = "57896044618658097711785492504343953926634992332820282019728792003956564819949" },
}

// ANCHOR: init
openvm::init!();
/* The init! macro will expand to the following
openvm_algebra_guest::moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141",
    "57896044618658097711785492504343953926634992332820282019728792003956564819949",
}

// have to implement Field for Edwards25519Coord because moduli_declare! only implements IntMod
impl Field for Edwards25519Coord {
    const ZERO: Self = <Self as IntMod>::ZERO;
    const ONE: Self = <Self as IntMod>::ONE;

    type SelfRef<'a> = &'a Self;

    fn double_assign(&mut self) {
        IntMod::double_assign(self);
    }

    fn square_assign(&mut self) {
        IntMod::square_assign(self);
    }
}

// a = 57896044618658097711785492504343953926634992332820282019728792003956564819948
// d = 37095705934669439343138083508754565189542113879843219016388785533085940283555
// encoded in little endian, 32 limbs of 8 bits each
const CURVE_A: Edwards25519Coord = Edwards25519Coord::from_const_bytes([
    236, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 127,
]);
const CURVE_D: Edwards25519Coord = Edwards25519Coord::from_const_bytes([
    163, 120, 89, 19, 202, 77, 235, 117, 171, 216, 65, 65, 77, 10, 112, 0, 152, 232, 121, 119, 121,
    64, 199, 140, 115, 254, 111, 43, 238, 108, 3, 82,
]);

// Note that we are defining the Edwards25519 curve for illustrative purposes only.
// In practice, we would use the ed25519 module which defines the Edwards25519 curve for us.
openvm_ecc_guest::te_macros::te_declare! {
    Edwards25519Point {
        mod_type = Edwards25519Coord,
        a = CURVE_A,
        d = CURVE_D
    }
}

openvm_ecc_guest::te_macros::te_init! {
    Edwards25519Point,
}

openvm_ecc_guest::sw_macros::sw_init! {
    Secp256k1Point,
}
*/
// ANCHOR_END: init

// ANCHOR: main
pub fn main() {
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

    #[allow(clippy::op_ref)]
    let _p3 = &p1 + &p2;

    let x1 = Edwards25519Coord::from_be_bytes(&hex!(
        "216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A"
    ));
    let y1 = Edwards25519Coord::from_be_bytes(&hex!(
        "6666666666666666666666666666666666666666666666666666666666666658"
    ));
    let p1 = Edwards25519Point::from_xy(x1, y1).unwrap();

    let x2 = Edwards25519Coord::from_u32(2);
    let y2 = Edwards25519Coord::from_be_bytes(&hex!(
        "1A43BF127BDDC4D71FF910403C11DDB5BA2BCDD2815393924657EF111E712631"
    ));
    let p2 = Edwards25519Point::from_xy(x2, y2).unwrap();

    #[allow(clippy::op_ref)]
    let _p3 = &p1 + &p2;
}
// ANCHOR_END: main
