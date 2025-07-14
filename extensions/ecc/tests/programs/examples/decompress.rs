#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::ops::Neg;
extern crate alloc;

use hex_literal::hex;
use openvm::io::read_vec;
use openvm_ecc_guest::{
    algebra::IntMod,
    weierstrass::{FromCompressed, WeierstrassPoint},
    Group,
};
use openvm_k256::{Secp256k1Coord, Secp256k1Point};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    // a prime that is 5 mod 8
    Fp5mod8 { modulus = "115792089237316195423570985008687907853269984665640564039457584007913129639501" },
    // a prime that is 1 mod 4
    Fp1mod4 { modulus = "0xffffffffffffffffffffffffffffffff000000000000000000000001" },
}

// const CURVE_B_5MOD8: Fp5mod8 = Fp5mod8::from_const_u8(3);
const CURVE_B_5MOD8: Fp5mod8 = Fp5mod8::from_const_u8(6);

const CURVE_A_1MOD4: Fp1mod4 = Fp1mod4::from_const_bytes(hex!(
    "FEFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000"
));
const CURVE_B_1MOD4: Fp1mod4 = Fp1mod4::from_const_bytes(hex!(
    "B4FF552343390B27BAD8BFD7B7B04450563241F5ABB3040C850A05B400000000"
));

openvm_ecc_sw_macros::sw_declare! {
    CurvePoint5mod8 {
        mod_type = Fp5mod8,
        b = CURVE_B_5MOD8,
    },
    CurvePoint1mod4 {
        mod_type = Fp1mod4,
        a = CURVE_A_1MOD4,
        b = CURVE_B_1MOD4,
    },
}

openvm::init!("openvm_init_decompress_k256.rs");

// test decompression under an honest host
pub fn main() {
    let bytes = read_vec();
    let x = Secp256k1Coord::from_le_bytes_unchecked(&bytes[..32]);
    let y = Secp256k1Coord::from_le_bytes_unchecked(&bytes[32..64]);
    let rec_id = y.as_le_bytes()[0] & 1;

    test_possible_decompression::<Secp256k1Point>(&x, &y, rec_id);
    // x = 5 is not on the x-coordinate of any point on the Secp256k1 curve
    test_impossible_decompression::<Secp256k1Point>(&Secp256k1Coord::from_u8(5), rec_id);

    let x = Fp5mod8::from_le_bytes_unchecked(&bytes[64..96]);
    let y = Fp5mod8::from_le_bytes_unchecked(&bytes[96..128]);
    let rec_id = y.as_le_bytes()[0] & 1;

    test_possible_decompression::<CurvePoint5mod8>(&x, &y, rec_id);
    // x = 0 is not on the x-coordinate of any point on the CurvePoint5mod8 curve
    test_impossible_decompression::<CurvePoint5mod8>(&Fp5mod8::ZERO, rec_id);
    // this x is such that y^2 = x^3 + 6 = 0
    // we want to test the case where y^2 = 0 and rec_id = 1
    let x = Fp5mod8::from_le_bytes_unchecked(&hex!(
        "d634a701c3b9b8cbf7797988be3953b442863b74d2d5c4d5f1a9de3c0c256d90"
    ));
    test_possible_decompression::<CurvePoint5mod8>(&x, &Fp5mod8::ZERO, 0);
    test_impossible_decompression::<CurvePoint5mod8>(&x, 1);

    let x = Fp1mod4::from_le_bytes_unchecked(&bytes[128..160]);
    let y = Fp1mod4::from_le_bytes_unchecked(&bytes[160..192]);
    let rec_id = y.as_le_bytes()[0] & 1;

    test_possible_decompression::<CurvePoint1mod4>(&x, &y, rec_id);
    // x = 1 is not on the x-coordinate of any point on the CurvePoint1mod4 curve
    test_impossible_decompression::<CurvePoint1mod4>(&Fp1mod4::from_u8(1), rec_id);
}

fn test_possible_decompression<P: WeierstrassPoint + FromCompressed<P::Coordinate>>(
    x: &P::Coordinate,
    y: &P::Coordinate,
    rec_id: u8,
) {
    let p = P::decompress(x.clone(), &rec_id).unwrap();
    assert_eq!(p.x(), x);
    assert_eq!(p.y(), y);
}

fn test_impossible_decompression<P: WeierstrassPoint + FromCompressed<P::Coordinate>>(
    x: &P::Coordinate,
    rec_id: u8,
) {
    let p = P::decompress(x.clone(), &rec_id);
    assert!(p.is_none());
}
