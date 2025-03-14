#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::ops::Neg;
extern crate alloc;

use openvm::io::read_vec;
use openvm_ecc_guest::{
    algebra::{Field, IntMod},
    k256::{Secp256k1Coord, Secp256k1Point},
    weierstrass::{DecompressionHint, FromCompressed, WeierstrassPoint},
    Group,
};

openvm::entry!(main);

openvm_algebra_moduli_macros::moduli_declare! {
    // a prime that is 1 mod 4
    Fp { modulus = "115792089237316195423570985008687907853269984665640564039457584007913129639501" },
}

openvm_algebra_moduli_macros::moduli_init! {
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F", // Secp256k1 modulus
    "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141", // Secp256k1 scalar
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
// Wrapper to override hint_decompress
#[allow(dead_code)] // clippy complains that the field is never read
struct Secp256k1PointWrapper(Secp256k1Point);

impl FromCompressed<Secp256k1Coord> for Secp256k1PointWrapper {
    // copied from Secp256k1Point::decompress implementation in sw-macros
    fn decompress(x: Secp256k1Coord, rec_id: &u8) -> Option<Self> {
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

    // override hint_decompress to return a dummy value
    #[allow(unused_variables)]
    fn hint_decompress(
        _x: &Secp256k1Coord,
        rec_id: &u8,
    ) -> Option<DecompressionHint<Secp256k1Coord>> {
        #[cfg(not(target_os = "zkvm"))]
        {
            unimplemented!()
        }
        #[cfg(target_os = "zkvm")]
        {
            // allow testing both possible and impossible hints
            if *rec_id & 1 == 0 {
                Some(DecompressionHint {
                    possible: false,
                    sqrt: Secp256k1Coord::from_u32(0),
                })
            } else {
                Some(DecompressionHint {
                    possible: true,
                    sqrt: Secp256k1Coord::from_u32(0),
                })
            }
        }
    }
}
impl Secp256k1PointWrapper {
    // copied from Secp256k1Point::honest_host_decompress implementation in sw-macros
    fn honest_host_decompress(x: &Secp256k1Coord, rec_id: &u8) -> Option<Option<Self>> {
        let hint = Secp256k1PointWrapper::hint_decompress(x, rec_id)?;

        if hint.possible {
            // ensure y < modulus
            hint.sqrt.assert_unique();

            if hint.sqrt.as_le_bytes()[0] & 1 != *rec_id & 1 {
                None
            } else {
                let ret = Secp256k1Point::from_xy_nonidentity(x.clone(), hint.sqrt)?;
                Some(Some(Secp256k1PointWrapper(ret)))
            }
        } else {
            // ensure sqrt < modulus
            hint.sqrt.assert_unique();

            let alpha = (x * x * x) + (x * &Secp256k1Point::CURVE_A) + &Secp256k1Point::CURVE_B;
            if &hint.sqrt * &hint.sqrt == alpha * Secp256k1Point::get_non_qr() {
                Some(None)
            } else {
                None
            }
        }
    }
}

// struct to override hint_decompress
#[allow(dead_code)] // clippy complains that the field is never read
struct MyCurvePointWrapper(MyCurvePoint);

impl FromCompressed<<MyCurvePoint as WeierstrassPoint>::Coordinate> for MyCurvePointWrapper {
    // copied from MyCurvePoint::decompress implementation in sw-macros
    fn decompress(x: <MyCurvePoint as WeierstrassPoint>::Coordinate, rec_id: &u8) -> Option<Self> {
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

    // override hint_decompress to return a dummy value
    #[allow(unused_variables)]
    fn hint_decompress(
        _x: &<MyCurvePoint as WeierstrassPoint>::Coordinate,
        rec_id: &u8,
    ) -> Option<DecompressionHint<<MyCurvePoint as WeierstrassPoint>::Coordinate>> {
        #[cfg(not(target_os = "zkvm"))]
        {
            unimplemented!()
        }
        #[cfg(target_os = "zkvm")]
        {
            if *rec_id & 1 == 0 {
                Some(DecompressionHint {
                    possible: false,
                    sqrt: <MyCurvePoint as WeierstrassPoint>::Coordinate::from_u32(0),
                })
            } else {
                Some(DecompressionHint {
                    possible: true,
                    sqrt: <MyCurvePoint as WeierstrassPoint>::Coordinate::from_u32(0),
                })
            }
        }
    }
}
impl MyCurvePointWrapper {
    fn honest_host_decompress(
        x: &<MyCurvePoint as WeierstrassPoint>::Coordinate,
        rec_id: &u8,
    ) -> Option<Option<Self>> {
        let hint = MyCurvePointWrapper::hint_decompress(x, rec_id)?;

        if hint.possible {
            // ensure proof fails if y >= modulus
            hint.sqrt.assert_unique();

            if hint.sqrt.as_le_bytes()[0] & 1 != *rec_id & 1 {
                None
            } else {
                let ret = MyCurvePoint::from_xy_nonidentity(x.clone(), hint.sqrt)?;
                Some(Some(MyCurvePointWrapper(ret)))
            }
        } else {
            // ensure proof fails if sqrt * sqrt != alpha * non_qr
            hint.sqrt.assert_unique();

            let alpha = (x * x * x) + (x * &MyCurvePoint::CURVE_A) + &MyCurvePoint::CURVE_B;
            if &hint.sqrt * &hint.sqrt == alpha * MyCurvePoint::get_non_qr() {
                Some(None)
            } else {
                None
            }
        }
    }
}

// Check that decompress enters an infinite loop when hint_decompress returns an incorrect value.
pub fn main() {
    setup_0();
    setup_2();
    setup_all_curves();

    let bytes = read_vec();

    test_p_3_mod_4(&bytes[..32], &bytes[32..64]);
    test_p_1_mod_4(&bytes[64..96], &bytes[96..128]);
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

// MyCurvePoint modulus is 1 mod 4
#[allow(unused_variables)]
fn test_p_1_mod_4(x: &[u8], y: &[u8]) {
    let x = <MyCurvePoint as WeierstrassPoint>::Coordinate::from_le_bytes(x);
    let _ = <MyCurvePoint as WeierstrassPoint>::Coordinate::from_le_bytes(y);

    #[cfg(feature = "test_mycurve_possible")]
    let p = MyCurvePointWrapper::decompress(x.clone(), &1);
    #[cfg(feature = "test_mycurve_impossible")]
    let p = MyCurvePointWrapper::decompress(x.clone(), &0);
}
