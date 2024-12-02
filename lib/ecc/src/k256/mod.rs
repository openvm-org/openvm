use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use axvm_algebra::IntMod;
use elliptic_curve::{
    sec1::{Coordinates, EncodedPoint, ModulusSize},
    Curve,
};
use hex_literal::hex;
use k256::Secp256k1;
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom1Funct3, SwBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

use super::group::{CyclicGroup, Group};
use crate::sw::IntrinsicCurve;

axvm::moduli_declare! {
    Secp256k1Coord { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F" },
    Secp256k1Scalar { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141" },
}

axvm::sw_setup! {
    Secp256k1Point { mod_type = Secp256k1Coord },
}

impl CyclicGroup for Secp256k1Point {
    const GENERATOR: Self = Secp256k1Point {
        x: Secp256k1Coord::from_const_bytes(hex!(
            "9817F8165B81F259D928CE2DDBFC9B02070B87CE9562A055ACBBDCF97E66BE79"
        )),
        y: Secp256k1Coord::from_const_bytes(hex!(
            "B8D410FB8FD0479C195485A648B417FDA808110EFCFBA45D65C4A32677DA3A48"
        )),
    };
    const NEG_GENERATOR: Self = Secp256k1Point {
        x: Secp256k1Coord::from_const_bytes(hex!(
            "9817F8165B81F259D928CE2DDBFC9B02070B87CE9562A055ACBBDCF97E66BE79"
        )),
        y: Secp256k1Coord::from_const_bytes(hex!(
            "7727EF046F2FB863E6AB7A59B74BE80257F7EEF103045BA29A3B5CD98825C5B7"
        )),
    };
}

impl IntrinsicCurve for Secp256k1 {
    type Scalar = Secp256k1Scalar;
    type Point = Secp256k1Point;
}
