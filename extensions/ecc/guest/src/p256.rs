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
            "96c298d84539a1f4a033eb2d817d0377f240a463e5e6bcf847422ce1f2d1176b"
        )),
        y: P256Coord::from_const_bytes(hex!(
            "f551bf376840b6cbce5e316b5733ce2b169e0f7c4aebe78e9b7f1afee242e34f"
        )),
    };
    const NEG_GENERATOR: Self = P256Point {
        x: P256Coord::from_const_bytes(hex!(
            "96c298d84539a1f4a033eb2d817d0377f240a463e5e6bcf847422ce1f2d1176b"
        )),
        y: P256Coord::from_const_bytes(hex!(
            "0aae40c897bf493431a1ce94a9cc31d4e961f083b51418716580e5011cbd1cb0"
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
