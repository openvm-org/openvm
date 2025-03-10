#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{
    msm,
    p256::{P256Coord, P256Point, P256Scalar},
    weierstrass::WeierstrassPoint,
    CyclicGroup, Group,
};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_init! {
    "0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff",
    "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"
}

openvm_ecc_sw_macros::sw_init! {
    P256Point,
}

pub fn main() {
    setup_all_moduli();
    setup_all_curves();

    // Sample points got from https://asecuritysite.com/ecc/p256p
    let x1 = P256Coord::from_u32(5);
    let y1 = P256Coord::from_le_bytes(&hex!(
        "ccfb4832085c4133c5a3d9643c50ca11de7a8199ce3b91fe061858aab9439245"
    ));
    let p1 = P256Point::from_xy(x1.clone(), y1.clone()).unwrap();
    let x2 = P256Coord::from_u32(6);
    let y2 = P256Coord::from_le_bytes(&hex!(
        "cb23828228510d22e9c0e70fb802d1dc47007233e5856946c20a25542c4cb236"
    ));
    let p2 = P256Point::from_xy(x2.clone(), y2.clone()).unwrap();

    // Generic add can handle equal or unequal points.
    let p3 = &p1 + &p2;
    let p4 = &p2 + &p2;

    // Add assign and double assign
    let mut sum = P256Point::from_xy(x1, y1).unwrap();
    sum += &p2;
    if sum.x() != p3.x() || sum.y() != p3.y() {
        panic!();
    }
    let mut double = P256Point::from_xy(x2, y2).unwrap();
    double.double_assign();
    if double.x() != p4.x() || double.y() != p4.y() {
        panic!();
    }

    // Test generator
    let (gen_x, gen_y) = P256Point::GENERATOR.into_coords();
    let generator = P256Point::from_xy(gen_x, gen_y).unwrap();
    let (neg_x, neg_y) = P256Point::NEG_GENERATOR.into_coords();
    let neg_generator = P256Point::from_xy(neg_x, neg_y).unwrap();
}
