// Fork of RustCrypto's p256 crate https://docs.rs/p256/latest/p256/
// that uses zkvm instructions

#![no_std]
extern crate alloc;

use elliptic_curve::{
    bigint::U256, consts::U32, point::PointCompression, Curve, CurveArithmetic, PrimeCurve,
};

mod coord;
mod internal;
mod point;
mod scalar;

#[cfg(feature = "ecdsa-core")]
pub mod ecdsa;

// Needs to be public so that the `sw_init` macro can access it
pub use internal::{P256Coord, P256Point, P256Scalar};

// -- Define the ZST for implementing the elliptic curve traits --
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, PartialOrd, Ord)]
pub struct NistP256;

// --- Implement the Curve trait on P256 ---

/// Order of the P256 elliptic curve in hexadecimal.
const ORDER_HEX: &str = "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";

/// Order of the P256 elliptic curve.
const ORDER: U256 = U256::from_be_hex(ORDER_HEX);

impl Curve for NistP256 {
    /// 32-byte serialized field elements.
    type FieldBytesSize = U32;

    // Perf: Use the U256 type from openvm_ruint here
    type Uint = U256;

    /// Curve order.
    const ORDER: U256 = ORDER;
}

impl PrimeCurve for NistP256 {}

impl CurveArithmetic for NistP256 {
    type AffinePoint = P256Point;
    /// The `ProjectivePoint` type is still internally represented as an affine point.
    type ProjectivePoint = P256Point;
    type Scalar = P256Scalar;
}

impl PointCompression for NistP256 {
    /// P256 points are typically uncompressed.
    const COMPRESS_POINTS: bool = false;
}

/// SEC1-encoded P256 curve point.
pub type EncodedPoint = elliptic_curve::sec1::EncodedPoint<NistP256>;
