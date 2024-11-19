use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use axvm_algebra::{DivUnsafe, IntMod};
use elliptic_curve::{
    sec1::{Coordinates, EncodedPoint, ModulusSize},
    Curve,
};
use hex_literal::hex;
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom1Funct3, ModArithBaseFunct7, SwBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

use super::group::{CyclicGroup, Group};

// TODO: consider consolidate with AffineCoords. Also separate encoding and x/y.
pub trait SwPoint: Group {
    type Coordinate: IntMod;

    // Ref: https://docs.rs/elliptic-curve/latest/elliptic_curve/sec1/index.html
    // Note: sec1 bytes are in big endian.
    fn from_encoded_point<C: Curve>(p: &EncodedPoint<C>) -> Self
    where
        C::FieldBytesSize: ModulusSize;

    // TODO: I(lunkai) tried to do to_encoded_point, but that requires the IntMod
    // to integrate with ModulusSize which is very annoying. So I just gave up for now and use bytes.
    // Note: sec1 bytes are in big endian.
    fn to_sec1_bytes(&self, is_compressed: bool) -> Vec<u8>;

    fn x(&self) -> Self::Coordinate;
    fn y(&self) -> Self::Coordinate;
}

axvm::moduli_setup! {
    Secp256k1Coord = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F";
    Secp256k1Scalar = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141";
}

axvm::sw_setup! {
    Secp256k1Point = Secp256k1Coord;
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
