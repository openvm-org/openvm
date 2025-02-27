use core::ops::{Add, Neg};

use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use lazy_static::lazy_static;
#[cfg(not(target_os = "zkvm"))]
use num_bigint::BigUint;
use openvm_algebra_guest::{Field, IntMod};

use super::group::{CyclicGroup, Group};
use crate::weierstrass::{CachedMulTable, IntrinsicCurve};

#[cfg(not(target_os = "zkvm"))]
lazy_static! {
    pub static ref P256_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff"
    ));
    pub static ref P256_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"
    ));
}

openvm_algebra_moduli_macros::moduli_declare! {
    P256Coord { modulus = "0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff" },
    P256Scalar { modulus = "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551" },
}

pub const P256_NUM_LIMBS: usize = 32;
pub const P256_LIMB_BITS: usize = 8;
pub const P256_BLOCK_SIZE: usize = 32;
// from_const_bytes is little endian
pub const CURVE_A: P256Coord = P256Coord::from_const_bytes(hex!(
    "fcffffffffffffffffffffff00000000000000000000000001000000ffffffff"
));
pub const CURVE_B: P256Coord = P256Coord::from_const_bytes(hex!(
    "4b60d2273e3cce3bf6b053ccb0061d65bc86987655bdebb3e7933aaad835c65a"
));

openvm_ecc_sw_macros::sw_declare! {
    P256Point { mod_type = P256Coord, a = CURVE_A, b = CURVE_B },
}

impl Field for P256Coord {
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

impl CyclicGroup for P256Point {
    const GENERATOR: Self = P256Point {
        x: P256Coord::from_const_bytes(hex!(
            "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
        )),
        y: P256Coord::from_const_bytes(hex!(
            "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"
        )),
    };
    const NEG_GENERATOR: Self = P256Point {
        x: P256Coord::from_const_bytes(hex!(
            "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
        )),
        y: P256Coord::from_const_bytes(hex!(
            "b01cbd1c01e58065711814b583f061e9d431cca994cea1313449bf97c840ae0a"
        )),
    };
}

impl IntrinsicCurve for p256::NistP256 {
    type Scalar = P256Scalar;
    type Point = P256Point;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
    where
        for<'a> &'a Self::Point: Add<&'a Self::Point, Output = Self::Point>,
    {
        if coeffs.len() < 25 {
            let table = CachedMulTable::<Self>::new_with_prime_order(bases, 4);
            table.windowed_mul(coeffs)
        } else {
            crate::msm(coeffs, bases)
        }
    }
}
