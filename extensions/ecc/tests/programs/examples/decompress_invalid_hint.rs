#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::ops::{Mul, Neg};
extern crate alloc;

use hex_literal::hex;
use openvm::io::read_vec;
use openvm_ecc_guest::{
    algebra::{Field, IntMod},
    k256::{Secp256k1Coord, Secp256k1Point},
    weierstrass::{DecompressionHint, FromCompressed, WeierstrassPoint},
    Group,
};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    // a prime that is 5 mod 8
    Fp5mod8 { modulus = "115792089237316195423570985008687907853269984665640564039457584007913129639501" },
    // a prime that is 1 mod 4
    Fp1mod4 { modulus = "0xffffffffffffffffffffffffffffffff000000000000000000000001" },
}

const CURVE_B_5MOD8: Fp5mod8 = Fp5mod8::from_const_u8(3);

impl Field for Fp5mod8 {
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

const CURVE_A_1MOD4: Fp1mod4 = Fp1mod4::from_const_bytes(hex!(
    "FEFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000"
));
const CURVE_B_1MOD4: Fp1mod4 = Fp1mod4::from_const_bytes(hex!(
    "B4FF552343390B27BAD8BFD7B7B04450563241F5ABB3040C850A05B400000000"
));

impl Field for Fp1mod4 {
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

openvm::init!("openvm_init_decompress_invalid_hint.rs");

trait NonQr<P: WeierstrassPoint> {
    fn get_non_qr() -> &'static P::Coordinate;
}

impl NonQr<Secp256k1Point> for Secp256k1Point {
    fn get_non_qr() -> &'static <Secp256k1Point as WeierstrassPoint>::Coordinate {
        Secp256k1Point::get_non_qr()
    }
}

impl NonQr<CurvePoint5mod8> for CurvePoint5mod8 {
    fn get_non_qr() -> &'static <CurvePoint5mod8 as WeierstrassPoint>::Coordinate {
        CurvePoint5mod8::get_non_qr()
    }
}

impl NonQr<CurvePoint1mod4> for CurvePoint1mod4 {
    fn get_non_qr() -> &'static <CurvePoint1mod4 as WeierstrassPoint>::Coordinate {
        CurvePoint1mod4::get_non_qr()
    }
}

// Wrapper to override hint_decompress
struct CurvePointWrapper<P: WeierstrassPoint>(P);

// Implement FromCompressed generically
impl<P: WeierstrassPoint> FromCompressed<P::Coordinate> for CurvePointWrapper<P>
where
    P: WeierstrassPoint + NonQr<P>,
    P::Coordinate: IntMod + 'static,
    for<'a> &'a P::Coordinate: Mul<&'a P::Coordinate, Output = P::Coordinate>,
{
    fn decompress(x: P::Coordinate, rec_id: &u8) -> Option<Self> {
        match Self::honest_host_decompress(&x, rec_id) {
            // successfully decompressed
            Some(Some(ret)) => Some(ret),
            // successfully proved that the point cannot be decompressed
            Some(None) => None,
            None => loop {
                openvm::io::println(
                    "ERROR: Decompression hint is invalid. Entering infinite loop.",
                );
            },
        }
    }

    #[allow(unused_variables)]
    fn hint_decompress(
        _x: &P::Coordinate,
        rec_id: &u8,
    ) -> Option<DecompressionHint<P::Coordinate>> {
        #[cfg(not(target_os = "zkvm"))]
        {
            unimplemented!()
        }
        #[cfg(target_os = "zkvm")]
        {
            // Test both possible and impossible hints
            if *rec_id & 1 == 0 {
                Some(DecompressionHint {
                    possible: false,
                    sqrt: P::Coordinate::from_u32(0),
                })
            } else {
                Some(DecompressionHint {
                    possible: true,
                    sqrt: P::Coordinate::from_u32(0),
                })
            }
        }
    }
}

impl<P: WeierstrassPoint> CurvePointWrapper<P>
where
    P: WeierstrassPoint + NonQr<P>,
    P::Coordinate: IntMod + 'static,
    for<'a> &'a P::Coordinate: Mul<&'a P::Coordinate, Output = P::Coordinate>,
{
    // copied from Secp256k1Point::honest_host_decompress implementation in sw-macros
    fn honest_host_decompress(x: &P::Coordinate, rec_id: &u8) -> Option<Option<Self>> {
        let hint = Self::hint_decompress(x, rec_id)?;

        if hint.possible {
            // ensure y < modulus
            hint.sqrt.assert_reduced();

            if hint.sqrt.as_le_bytes()[0] & 1 != *rec_id & 1 {
                None
            } else {
                let ret = P::from_xy_nonidentity(x.clone(), hint.sqrt)?;
                Some(Some(CurvePointWrapper(ret)))
            }
        } else {
            // ensure sqrt < modulus
            hint.sqrt.assert_reduced();

            let alpha = (x * x * x) + (x * &P::CURVE_A) + &P::CURVE_B;
            if &hint.sqrt * &hint.sqrt == alpha * P::get_non_qr() {
                Some(None)
            } else {
                None
            }
        }
    }
}

// Create type aliases for each specific curve
#[allow(dead_code)]
type Secp256k1PointWrapper = CurvePointWrapper<Secp256k1Point>;
#[allow(dead_code)]
type CurvePoint5mod8Wrapper = CurvePointWrapper<CurvePoint5mod8>;
#[allow(dead_code)]
type CurvePoint1mod4Wrapper = CurvePointWrapper<CurvePoint1mod4>;

// Check that decompress enters an infinite loop when hint_decompress returns an incorrect value.
pub fn main() {
    let bytes = read_vec();

    test_p_3_mod_4(&bytes[..32], &bytes[32..64]);
    test_p_5_mod_8(&bytes[64..96], &bytes[96..128]);
    test_p_1_mod_4(&bytes[128..160], &bytes[160..192]);
}

// Secp256k1 modulus is 3 mod 4
#[allow(unused_variables)]
fn test_p_3_mod_4(x: &[u8], y: &[u8]) {
    let x = Secp256k1Coord::from_le_bytes(x);
    let _ = Secp256k1Coord::from_le_bytes(y);

    #[cfg(feature = "test_secp256k1_possible")]
    let p = Secp256k1PointWrapper::decompress(x.clone(), &1);
    #[cfg(feature = "test_secp256k1_impossible")]
    let p = Secp256k1PointWrapper::decompress(x.clone(), &0);
}

// CurvePoint5mod8 modulus is 5 mod 8
#[allow(unused_variables)]
fn test_p_5_mod_8(x: &[u8], y: &[u8]) {
    let x = <CurvePoint5mod8 as WeierstrassPoint>::Coordinate::from_le_bytes(x);
    let _ = <CurvePoint5mod8 as WeierstrassPoint>::Coordinate::from_le_bytes(y);

    #[cfg(feature = "test_curvepoint5mod8_possible")]
    let p = CurvePoint5mod8Wrapper::decompress(x.clone(), &1);
    #[cfg(feature = "test_curvepoint5mod8_impossible")]
    let p = CurvePoint5mod8Wrapper::decompress(x.clone(), &0);
}

// CurvePoint1mod4 modulus is 1 mod 4
#[allow(unused_variables)]
fn test_p_1_mod_4(x: &[u8], y: &[u8]) {
    let x = <CurvePoint1mod4 as WeierstrassPoint>::Coordinate::from_le_bytes(x);
    let _ = <CurvePoint1mod4 as WeierstrassPoint>::Coordinate::from_le_bytes(y);

    #[cfg(feature = "test_curvepoint1mod4_possible")]
    let p = CurvePoint1mod4Wrapper::decompress(x.clone(), &1);
    #[cfg(feature = "test_curvepoint1mod4_impossible")]
    let p = CurvePoint1mod4Wrapper::decompress(x.clone(), &0);
}
