use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul};

use openvm_algebra_guest::{IntMod, Reduce};

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

    /// Hazmat: Assumes p1 != +- p2 and p != identity and p2 != identity.
    fn add_ne_nonidentity(p1: &Self, p2: &Self) -> Self;
    /// Hazmat: Assumes self != +- p2 and self != identity and p2 != identity.
    fn add_ne_assign_nonidentity(&mut self, p2: &Self);
    /// Hazmat: Assumes p != identity and 2 * p != identity.
    fn double_nonidentity(p: &Self) -> Self;
    /// Hazmat: Assumes self != identity and 2 * self != identity.
    fn double_assign_nonidentity(&mut self);

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

/// A trait for elliptic curves that bridges the openvm types and external types with CurveArithmetic etc.
/// Implement this for external curves with corresponding openvm point and scalar types.
pub trait IntrinsicCurve {
    type Scalar: IntMod + Reduce;
    type Point: WeierstrassPoint + CyclicGroup;

    /// Multi-scalar multiplication. The default implementation may be
    /// replaced by specialized implementations that use properties of the curve
    /// (e.g., if the curve order is prime).
    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
    where
        for<'a> &'a Self::Point: Add<&'a Self::Point, Output = Self::Point>,
    {
        super::msm(coeffs, bases)
    }
}

// MSM using preprocessed table (windowed method)
// Reference: modified from https://github.com/arkworks-rs/algebra/blob/master/ec/src/scalar_mul/mod.rs
//
// We specialize to Weierstrass curves and further make optimizations for when the curve order is prime.

/// Cached precomputations of scalar multiples of several base points.
/// - `window_bits` is the window size used for the precomputation
/// - `max_scalar_bits` is the maximum size of the scalars that will be multiplied
/// - `table` is the precomputed table
pub struct CachedMulTable<'a, C: IntrinsicCurve> {
    /// Window bits. Must be > 0.
    /// For alignment, we currently require this to divide 8 (bits in a byte).
    pub window_bits: usize,
    pub bases: &'a [C::Point],
    /// `table[i][j] = (j + 2) * bases[i]` for `j + 2 < 2 ** window_bits`
    table: Vec<Vec<C::Point>>,
    /// Needed to return reference to the identity point.
    identity: C::Point,
}

impl<'a, C: IntrinsicCurve> CachedMulTable<'a, C> {
    /// Constructor when the curve order is prime (so the group of curve points forms the scalar prime field).
    ///
    /// Assumes that `window_bits` is less than number of bits - 1 in the modulus
    /// of `C::Scalar`.
    pub fn new_with_prime_order(bases: &'a [C::Point], window_bits: usize) -> Self {
        assert!(window_bits > 0);
        let window_size = 1 << window_bits;
        let table = bases
            .iter()
            .map(|base| {
                if base.is_identity() {
                    vec![C::Point::IDENTITY; window_size - 2]
                } else {
                    let mut multiples = Vec::with_capacity(window_size - 2);
                    for _ in 0..window_size - 2 {
                        // Because curve order is prime, we are guaranteed that
                        // j * base != identity,
                        // j * base != +- base for j > 1,
                        // j * base + base != identity
                        let multiple = multiples
                            .last()
                            .map(|last| C::Point::add_ne_nonidentity(last, base))
                            .unwrap_or_else(|| C::Point::double_nonidentity(base));
                        multiples.push(multiple);
                    }
                    multiples
                }
            })
            .collect();

        Self {
            window_bits,
            bases,
            table,
            identity: C::Point::IDENTITY,
        }
    }

    fn get_multiple(&self, base_idx: usize, scalar: usize) -> &C::Point {
        if scalar == 0 {
            &self.identity
        } else if scalar == 1 {
            unsafe { self.bases.get_unchecked(base_idx) }
        } else {
            unsafe { self.table.get_unchecked(base_idx).get_unchecked(scalar - 2) }
        }
    }

    /// Computes `sum scalars[i] * bases[i]`.
    ///
    /// For implementation simplicity, currently only implemented when
    /// `window_bits` divides 8 (number of bits in a byte).
    pub fn windowed_mul(&self, scalars: &[C::Scalar]) -> C::Point {
        assert_eq!(8 % self.window_bits, 0);
        assert_eq!(scalars.len(), self.bases.len());
        let windows_per_byte = 8 / self.window_bits;

        let num_windows = C::Scalar::NUM_LIMBS * windows_per_byte;
        let mask = (1u8 << self.window_bits) - 1;

        // The current byte index (little endian) at the current step of the
        // windowed method, across all scalars.
        let mut limb_idx = C::Scalar::NUM_LIMBS;
        // The current bit (little endian) within the current byte of the windowed
        // method. The window will look at bits `bit_idx..bit_idx + window_bits`.
        // bit_idx will always be in range [0, 8)
        let mut bit_idx = 0;

        let mut res = C::Point::IDENTITY;
        for outer in 0..num_windows {
            if bit_idx == 0 {
                limb_idx -= 1;
                bit_idx = 8 - self.window_bits;
            } else {
                bit_idx -= self.window_bits;
            }

            if outer != 0 {
                for _ in 0..self.window_bits {
                    // Note: this handles identity
                    res.double_assign();
                }
            }
            for (base_idx, scalar) in scalars.iter().enumerate() {
                let scalar = (scalar.as_le_bytes()[limb_idx] >> bit_idx) & mask;
                let summand = self.get_multiple(base_idx, scalar as usize);
                // handles identity
                res.add_assign(summand);
            }
        }
        res
    }
}
