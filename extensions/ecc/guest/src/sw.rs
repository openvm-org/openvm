use alloc::vec::Vec;

use axvm_algebra_guest::{IntMod, Reduce};
use elliptic_curve::{
    sec1::{EncodedPoint, ModulusSize},
    Curve,
};

use super::group::{CyclicGroup, Group};

// TODO: consider consolidate with AffineCoords. Also separate encoding and x/y.
/// Short Weierstrass curve affine point.
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

/// A trait for elliptic curves that bridges the axvm types and external types with CurveArithmetic etc.
/// Implement this for external curves with corresponding axvm point and scalar types.
pub trait IntrinsicCurve {
    type Scalar: IntMod + Reduce;
    type Point: SwPoint + CyclicGroup;
}
