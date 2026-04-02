// Fork of RustCrypto's k256 crate https://docs.rs/k256/latest/k256/
// that uses zkvm instructions

#![no_std]
extern crate alloc;

use elliptic_curve::{consts::U32, point::PointCompression, Curve, CurveArithmetic, PrimeCurve};

mod coord;
mod internal;
mod point;
mod scalar;

#[cfg(feature = "ecdsa-core")]
pub mod ecdsa;

pub use elliptic_curve::{self, bigint::U256};
// Needs to be public so that the `sw_init` macro can access it
pub use internal::{
    Secp256k1Coord, Secp256k1Point, Secp256k1Point as AffinePoint,
    Secp256k1Point as ProjectivePoint, Secp256k1Scalar as Scalar, Secp256k1Scalar,
};

// -- Define the ZST for implementing the elliptic curve traits --
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, PartialOrd, Ord)]
pub struct Secp256k1;

// --- Implement the Curve trait on Secp256k1 ---

/// Order of the secp256k1 elliptic curve in hexadecimal.
const ORDER_HEX: &str = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";

/// Order of the secp256k1 elliptic curve.
const ORDER: U256 = U256::from_be_hex(ORDER_HEX);

impl Curve for Secp256k1 {
    /// 32-byte serialized field elements.
    type FieldBytesSize = U32;

    // Perf: Use the U256 type from openvm_ruint here
    type Uint = U256;

    /// Curve order.
    const ORDER: U256 = ORDER;
}

impl PrimeCurve for Secp256k1 {}

impl CurveArithmetic for Secp256k1 {
    type AffinePoint = AffinePoint;
    type ProjectivePoint = ProjectivePoint;
    type Scalar = Scalar;
}

impl PointCompression for Secp256k1 {
    /// secp256k1 points are typically compressed.
    const COMPRESS_POINTS: bool = true;
}

/// SEC1-encoded secp256k1 (K-256) curve point.
pub type EncodedPoint = elliptic_curve::sec1::EncodedPoint<Secp256k1>;
