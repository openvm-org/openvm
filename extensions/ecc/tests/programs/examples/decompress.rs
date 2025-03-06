#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::ops::Neg;
extern crate alloc;

use openvm::io::read_vec;
use openvm_ecc_guest::{
    algebra::{Field, IntMod},
    k256::{Secp256k1Coord, Secp256k1Point},
    weierstrass::{FromCompressed, WeierstrassPoint},
    Group,
};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    // a prime that is 1 mod 4
    Fp { modulus = "115792089237316195423570985008687907853269984665640564039457584007913129639501" },
}

openvm_algebra_moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F",
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141",
    "115792089237316195423570985008687907853269984665640564039457584007913129639501",
}

impl Field for Fp {
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

const MY_CURVE_B: Fp = Fp::from_const_u8(3);

openvm_ecc_sw_macros::sw_declare! {
    MyCurvePoint {
        mod_type = Fp,
        b = MY_CURVE_B,
    }
}

openvm_ecc_sw_macros::sw_init! {
    Secp256k1Point,
    MyCurvePoint,
}

// test decompression under an honest host
pub fn main() {
    setup_0();
    setup_2();
    setup_all_curves();

    let bytes = read_vec();
    let x = Secp256k1Coord::from_le_bytes(&bytes[..32]);
    let y = Secp256k1Coord::from_le_bytes(&bytes[32..64]);
    let rec_id = y.as_le_bytes()[0] & 1;

    test_possible_decompression::<Secp256k1Point>(&x, &y, rec_id);
    // x = 5 is not on the x-coordinate of any point on the Secp256k1 curve
    test_impossible_decompression_secp256k1(&Secp256k1Coord::from_u8(5), rec_id);

    let x = Fp::from_le_bytes(&bytes[64..96]);
    let y = Fp::from_le_bytes(&bytes[96..128]);
    let rec_id = y.as_le_bytes()[0] & 1;

    test_possible_decompression::<MyCurvePoint>(&x, &y, rec_id);
    // x = 3 is not on the x-coordinate of any point on the MyCurve curve
    test_impossible_decompression_mycurve(&Fp::from_u8(3), rec_id);
}

fn test_possible_decompression<P: WeierstrassPoint + FromCompressed<P::Coordinate>>(
    x: &P::Coordinate,
    y: &P::Coordinate,
    rec_id: u8,
) {
    let hint = P::hint_decompress(x, &rec_id).expect("hint should be well-formed");
    if hint.possible {
        assert_eq!(y, &hint.sqrt);
    } else {
        panic!("decompression should be possible");
    }

    let p = P::decompress(x.clone(), &rec_id).unwrap();
    assert_eq!(p.x(), x);
    assert_eq!(p.y(), y);
}

// The test_impossible_decompression_* functions cannot be combined into a single function with a const generic parameter
// since the get_non_qr() function is not part of the WeierstrassPoint trait.

fn test_impossible_decompression_mycurve(x: &Fp, rec_id: u8) {
    let hint = MyCurvePoint::hint_decompress(x, &rec_id).expect("hint should be well-formed");
    if hint.possible {
        panic!("decompression should be impossible");
    } else {
        let rhs = x * x * x
            + x * &<MyCurvePoint as WeierstrassPoint>::CURVE_A
            + &<MyCurvePoint as WeierstrassPoint>::CURVE_B;
        assert_eq!(&hint.sqrt * &hint.sqrt, rhs * MyCurvePoint::get_non_qr());
    }

    let p = MyCurvePoint::decompress(x.clone(), &rec_id);
    assert!(p.is_none());
}

fn test_impossible_decompression_secp256k1(x: &Secp256k1Coord, rec_id: u8) {
    let hint = Secp256k1Point::hint_decompress(x, &rec_id).expect("hint should be well-formed");
    if hint.possible {
        panic!("decompression should be impossible");
    } else {
        let rhs = x * x * x
            + x * &<Secp256k1Point as WeierstrassPoint>::CURVE_A
            + &<Secp256k1Point as WeierstrassPoint>::CURVE_B;
        assert_eq!(&hint.sqrt * &hint.sqrt, rhs * Secp256k1Point::get_non_qr());
    }

    let p = Secp256k1Point::decompress(x.clone(), &rec_id);
    assert!(p.is_none());
}
