use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use axvm_algebra::{DivUnsafe, IntMod};
use elliptic_curve::{
    sec1::{Coordinates, EncodedPoint, ModulusSize},
    Curve,
};
#[cfg(target_os = "zkvm")]
use {
    axvm_platform::constants::{Custom1Funct3, ModArithBaseFunct7, SwBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

use super::group::Group;

pub trait SwPoint: Group {
    type Coordinate: IntMod;

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

// Each curve has: (Name, GeneratorX, GeneratorY)
axvm::sw_setup! {
    Secp256k1Point = (
        Secp256k1Coord,
        "0x79be667e f9dcbbac 55a06295 ce870b07 029bfcdb 2dce28d9 59f2815b 16f81798",
        "0x483ada77 26a3c465 5da4fbfc 0e1108a8 fd17b448 a6855419 9c47d08f fb10d4b8",
    );
}
