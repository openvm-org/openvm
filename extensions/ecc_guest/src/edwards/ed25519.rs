extern crate alloc;

use core::ops::Add;

use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use lazy_static::lazy_static;
#[cfg(not(target_os = "zkvm"))]
use num_bigint::BigUint;
use openvm_algebra_guest::IntMod;
use openvm_edwards_guest::{CyclicGroup, IntrinsicCurve};

#[cfg(not(target_os = "zkvm"))]
lazy_static! {
    pub static ref ED25519_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED"
    ));
    pub static ref ED25519_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED"
    ));
    pub static ref ED25519_A: BigUint = BigUint::from_bytes_be(&hex!(
        "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEC"
    ));
    pub static ref ED25519_D: BigUint = BigUint::from_bytes_be(&hex!(
        "52036CEE2B6FFE738CC740797779E89800700A4D4141D8AB75EB4DCA135978A3"
    ));
}

openvm_algebra_guest::moduli_macros::moduli_declare! {
    Ed25519Coord { modulus = "0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED" },
    Ed25519Scalar { modulus = "0x1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED" },
}

pub const ED25519_NUM_LIMBS: usize = 32;
pub const ED25519_LIMB_BITS: usize = 8;
pub const ED25519_BLOCK_SIZE: usize = 32;
// from_const_bytes is little endian
pub const CURVE_A: Ed25519Coord = Ed25519Coord::from_const_bytes(hex!(
    "ECFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7F"
));
pub const CURVE_D: Ed25519Coord = Ed25519Coord::from_const_bytes(hex!(
    "A3785913CA4DEB75ABD841414D0A700098E879777940C78C73FE6F2BEE6C0352"
));

openvm_edwards_guest::te_macros::te_declare! {
    Ed25519Point { mod_type = Ed25519Coord, a = CURVE_A, d = CURVE_D },
}

impl CyclicGroup for Ed25519Point {
    // from_const_bytes is little endian
    const GENERATOR: Self = Ed25519Point {
        x: Ed25519Coord::from_const_bytes(hex!(
            "1AD5258F602D56C9B2A7259560C72C695CDCD6FD31E2A4C0FE536ECDD3366921"
        )),
        y: Ed25519Coord::from_const_bytes(hex!(
            "5866666666666666666666666666666666666666666666666666666666666666"
        )),
    };
    const NEG_GENERATOR: Self = Ed25519Point {
        x: Ed25519Coord::from_const_bytes([
            211, 42, 218, 112, 159, 210, 169, 54, 77, 88, 218, 106, 159, 56, 211, 150, 163, 35, 41,
            2, 206, 29, 91, 63, 1, 172, 145, 50, 44, 201, 150, 94,
        ]),
        y: Ed25519Coord::from_const_bytes(hex!(
            "5866666666666666666666666666666666666666666666666666666666666666"
        )),
    };
}

impl IntrinsicCurve for Ed25519Point {
    type Scalar = Ed25519Scalar;
    type Point = Ed25519Point;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
    where
        for<'a> &'a Self::Point: Add<&'a Self::Point, Output = Self::Point>,
    {
        if coeffs.len() < 25 {
            let table = openvm_edwards_guest::edwards::CachedMulTable::<Self>::new(bases, 4);
            table.windowed_mul(coeffs)
        } else {
            openvm_edwards_guest::msm(coeffs, bases)
        }
    }
}
