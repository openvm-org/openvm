use core::ops::Mul;

use axvm_algebra_guest::{IntMod, Reduce};

use super::group::{CyclicGroup, Group};

/// Short Weierstrass curve affine point.
pub trait WeierstrassPoint: Group {
    /// The `b` coefficient in the Weierstrass curve equation `y^2 = x^3 + a x + b`.
    const CURVE_B: Self::Coordinate;

    type Coordinate: IntMod;

    /// The concatenated `x, y` coordinates of the affine point, where
    /// coordinates are in little endian.
    ///
    /// **Warning**: The memory layout of `Self` is expected to pack
    /// `x` and `y` contigously with no unallocated space in between.
    fn as_le_bytes(&self) -> &[u8];

    /// Raw constructor without asserting point is on the curve.
    fn from_xy_unchecked(x: Self::Coordinate, y: Self::Coordinate) -> Self;
    fn into_coords(self) -> (Self::Coordinate, Self::Coordinate);
    fn x(&self) -> &Self::Coordinate;
    fn y(&self) -> &Self::Coordinate;
    fn x_mut(&mut self) -> &mut Self::Coordinate;
    fn y_mut(&mut self) -> &mut Self::Coordinate;

    fn from_xy(x: Self::Coordinate, y: Self::Coordinate) -> Option<Self>
    where
        for<'a> &'a Self::Coordinate: Mul<&'a Self::Coordinate, Output = Self::Coordinate>,
    {
        let lhs = &y * &y;
        let rhs = &x * &x * &x + &Self::CURVE_B;
        if lhs != rhs {
            return None;
        }
        Some(Self::from_xy_unchecked(x, y))
    }

    /// Given `x`-coordinate,
    ///
    /// ## Panics
    /// If the input is not a valid compressed point.
    /// The zkVM panics instead of returning an [Option] because this function
    /// can only guarantee correct behavior when decompression is possible,
    /// but the function cannot compute the boolean equal to true if and only
    /// if decompression is possible.
    // This is because we rely on a hint for the correct decompressed value
    // and then constrain its correctness. A malicious prover could hint
    // incorrectly, so there is no way to use a hint to prove that the input
    // **cannot** be decompressed.
    fn decompress(x: Self::Coordinate, rec_id: &u8) -> Self
    where
        for<'a> &'a Self::Coordinate: Mul<&'a Self::Coordinate, Output = Self::Coordinate>,
    {
        let y = Self::hint_decompress(&x, rec_id);
        // Must assert unique so we can check the parity
        y.assert_unique();
        assert_eq!(y.as_le_bytes()[0] & 1, *rec_id & 1);
        Self::from_xy(x, y).expect("decompressed point not on curve")
    }

    /// If it exists, hints the unique `y` coordinate that is less than `Coordinate::MODULUS`
    /// such that `(x, y)` is a point on the curve and `y` has parity equal to `rec_id`.
    /// If such `y` does not exist, undefined behavior.
    ///
    /// This is only a hint, and the returned `y` does not guarantee any of the above properties.
    /// They must be checked separately. Normal users should use `decompress` directly.
    fn hint_decompress(x: &Self::Coordinate, rec_id: &u8) -> Self::Coordinate;
}

/// A trait for elliptic curves that bridges the axvm types and external types with CurveArithmetic etc.
/// Implement this for external curves with corresponding axvm point and scalar types.
pub trait IntrinsicCurve {
    type Scalar: IntMod + Reduce;
    type Point: WeierstrassPoint + CyclicGroup;
}
