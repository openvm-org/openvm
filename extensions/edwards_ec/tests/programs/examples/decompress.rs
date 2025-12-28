#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::ops::Neg;
extern crate alloc;

use hex_literal::hex;
use openvm::io::read_vec;
use openvm_te_guest::{
    algebra::IntMod,
    ed25519::{Ed25519Coord, Ed25519Point},
    edwards::TwistedEdwardsPoint,
    FromCompressed, Group,
};

openvm::entry!(main);

openvm::init!("openvm_init_decompress_ed25519.rs");

// test decompression under an honest host
pub fn main() {
    let bytes = read_vec();

    let x = Ed25519Coord::from_le_bytes_unchecked(&bytes[..32]);
    let y = Ed25519Coord::from_le_bytes_unchecked(&bytes[32..64]);
    let rec_id = x.as_le_bytes()[0] & 1;
    test_possible_decompression::<Ed25519Point>(&x, &y, rec_id);
    // y = 2 is not on the y-coordinate of any point on the Ed25519 curve
    test_impossible_decompression::<Ed25519Point>(&Ed25519Coord::from_u8(2), rec_id);
}

fn test_possible_decompression<P: TwistedEdwardsPoint + FromCompressed<P::Coordinate>>(
    x: &P::Coordinate,
    y: &P::Coordinate,
    rec_id: u8,
) {
    let p = P::decompress(y.clone(), &rec_id).unwrap();
    assert_eq!(p.x(), x);
    assert_eq!(p.y(), y);
}

fn test_impossible_decompression<P: TwistedEdwardsPoint + FromCompressed<P::Coordinate>>(
    x: &P::Coordinate,
    rec_id: u8,
) {
    let p = P::decompress(x.clone(), &rec_id);
    assert!(p.is_none());
}
