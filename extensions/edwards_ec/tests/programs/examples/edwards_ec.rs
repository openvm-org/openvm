#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use hex_literal::hex;
use openvm_algebra_guest::moduli_macros::moduli_init;
use openvm_ecc_guest::edwards::{
    algebra::IntMod,
    ed25519::{Ed25519Coord, Ed25519Point},
    edwards::TwistedEdwardsPoint,
    te_macros::te_init,
    CyclicGroup, Group,
};

openvm::init!("openvm_init_edwards_ec_ed25519.rs");

openvm::entry!(main);

pub fn main() {
    // Base point of edwards25519
    let mut p1 = Ed25519Point::GENERATOR;

    // random point on edwards25519
    let x2 = Ed25519Coord::from_u32(2);
    let y2 = Ed25519Coord::from_be_bytes_unchecked(&hex!(
        "1A43BF127BDDC4D71FF910403C11DDB5BA2BCDD2815393924657EF111E712631"
    ));
    let mut p2 = Ed25519Point::from_xy(x2, y2).unwrap();

    // This is the sum of (x1, y1) and (x2, y2).
    let x3 = Ed25519Coord::from_be_bytes_unchecked(&hex!(
        "636C0B519B2C5B1E0D3BFD213F45AFD5DAEE3CECC9B68CF88615101BC78329E6"
    ));
    let y3 = Ed25519Coord::from_be_bytes_unchecked(&hex!(
        "704D8868CB335A7B609D04B9CD619511675691A78861F1DFF7A5EBC389C7EA92"
    ));

    // This is 2 * (x1, y1)
    let x4 = Ed25519Coord::from_be_bytes_unchecked(&hex!(
        "56B98CC045559AD2BBC45CAB58D842ECEE264DB9395F6014B772501B62BB7EE8"
    ));
    let y4 = Ed25519Coord::from_be_bytes_unchecked(&hex!(
        "1BCA918096D89C83A15105DF343DC9F7510494407750226DAC0A7620ACE77BEB"
    ));

    // Generic add can handle equal or unequal points.
    let p3 = &p1 + &p2;
    if p3.x() != &x3 || p3.y() != &y3 {
        panic!();
    }
    let p4 = &p2 + &p2;
    if p4.x() != &x4 || p4.y() != &y4 {
        panic!();
    }

    // Add assign and double assign
    p1 += &p2;
    if p1.x() != &x3 || p1.y() != &y3 {
        panic!();
    }
    p2.double_assign();
    if p2.x() != &x4 || p2.y() != &y4 {
        panic!();
    }
}
