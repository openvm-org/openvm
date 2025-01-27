use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use lazy_static::lazy_static;
#[cfg(not(target_os = "zkvm"))]
use num_bigint::BigUint;
use openvm_algebra_guest::{Field, IntMod};

use super::group::{CyclicGroup, Group};
use crate::IntrinsicCurve;

#[cfg(not(target_os = "zkvm"))]
lazy_static! {
    pub static ref Ed25519_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED"
    ));
    pub static ref Ed25519_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED"
    ));
    pub static ref Ed25519_A: BigUint = BigUint::from_bytes_be(&hex!(
        "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEC"
    ));
    pub static ref Ed25519_D: BigUint = BigUint::from_bytes_be(&hex!(
        "52036CEE2B6FFE738CC740797779E89800700A4D4141D8AB75EB4DCA135978A3"
    ));
}

openvm_algebra_moduli_setup::moduli_declare! {
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

openvm_ecc_te_setup::te_declare! {
    Ed25519Point { mod_type = Ed25519Coord, a = CURVE_A, d = CURVE_D },
}

impl Field for Ed25519Coord {
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
    // TODO: fix
    const NEG_GENERATOR: Self = Ed25519Point {
        x: Ed25519Coord::from_const_bytes(hex!(
            "1AD5258F602D56C9B2A7259560C72C695CDCD6FD31E2A4C0FE536ECDD3366921"
        )),
        y: Ed25519Coord::from_const_bytes(hex!(
            "5866666666666666666666666666666666666666666666666666666666666666"
        )),
    };
}

impl IntrinsicCurve for Ed25519Point {
    type Scalar = Ed25519Scalar;
    type Point = Ed25519Point;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point {
        // TODO: idk if this can be optimized
        openvm_ecc_guest::msm(coeffs, bases)
    }
}
