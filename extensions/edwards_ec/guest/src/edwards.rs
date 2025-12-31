use alloc::vec::Vec;
use core::ops::{AddAssign, Mul};

use openvm_algebra_guest::{Field, IntMod};

use crate::{Group, IntrinsicCurve};

pub trait TwistedEdwardsPoint: Sized {
    /// The `a` coefficient in the twisted Edwards curve equation `ax^2 + y^2 = 1 + d x^2 y^2`.
    const CURVE_A: Self::Coordinate;
    /// The `d` coefficient in the twisted Edwards curve equation `ax^2 + y^2 = 1 + d x^2 y^2`.
    const CURVE_D: Self::Coordinate;
    const IDENTITY: Self;

    type Coordinate: Field;

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

    fn add_impl(&self, p2: &Self) -> Self;

    #[inline(always)]
    fn from_xy(x: Self::Coordinate, y: Self::Coordinate) -> Option<Self>
    where
        for<'a> &'a Self::Coordinate: Mul<&'a Self::Coordinate, Output = Self::Coordinate>,
    {
        let lhs = Self::CURVE_A * &x * &x + &y * &y;
        let rhs = Self::CURVE_D * &x * &x * &y * &y + &Self::Coordinate::ONE;
        if lhs != rhs {
            return None;
        }
        Some(Self::from_xy_unchecked(x, y))
    }
}

/// Macro to generate a newtype wrapper for [AffinePoint](crate::AffinePoint)
/// that implements elliptic curve operations by using the underlying field operations according to
/// the [formulas](https://en.wikipedia.org/wiki/Twisted_Edwards_curve) for twisted Edwards curves.
///
/// The following imports are required:
/// ```rust
/// use core::ops::AddAssign;
///
/// use openvm_algebra_guest::{DivUnsafe, Field};
/// use openvm_weierstrass_guest::{edwards::TwistedEdwardsPoint, AffinePoint, Group};
/// ```
#[macro_export]
macro_rules! impl_te_affine {
    ($struct_name:ident, $field:ty, $a:expr, $d:expr) => {
        /// A newtype wrapper for [AffinePoint] that implements elliptic curve operations
        /// by using the underlying field operations according to the [formulas](https://en.wikipedia.org/wiki/Twisted_Edwards_curve) for twisted Edwards curves.
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $struct_name(AffinePoint<$field>);

        impl TwistedEdwardsPoint for $struct_name {
            const CURVE_A: $field = $a;
            const CURVE_D: $field = $d;
            const IDENTITY: Self = Self(AffinePoint::new(<$field>::ZERO, <$field>::ONE));

            type Coordinate = $field;

            /// SAFETY: assumes that [$field] has internal representation in little-endian.
            fn as_le_bytes(&self) -> &[u8] {
                unsafe {
                    &*core::ptr::slice_from_raw_parts(
                        self as *const Self as *const u8,
                        core::mem::size_of::<Self>(),
                    )
                }
            }
            fn from_xy_unchecked(x: Self::Coordinate, y: Self::Coordinate) -> Self {
                Self(AffinePoint::new(x, y))
            }
            fn into_coords(self) -> (Self::Coordinate, Self::Coordinate) {
                (self.0.x, self.0.y)
            }
            fn x(&self) -> &Self::Coordinate {
                &self.0.x
            }
            fn y(&self) -> &Self::Coordinate {
                &self.0.y
            }
            fn x_mut(&mut self) -> &mut Self::Coordinate {
                &mut self.0.x
            }
            fn y_mut(&mut self) -> &mut Self::Coordinate {
                &mut self.0.y
            }

            fn add_impl(&self, p2: &Self) -> Self {
                use ::openvm_algebra_guest::DivUnsafe;
                // For twisted Edwards curves:
                // x3 = (x1*y2 + y1*x2)/(1 + d*x1*x2*y1*y2)
                // y3 = (y1*y2 - a*x1*x2)/(1 - d*x1*x2*y1*y2)
                let x1y2 = self.x() * p2.y();
                let y1x2 = self.y() * p2.x();
                let x1x2 = self.x() * p2.x();
                let y1y2 = self.y() * p2.y();
                let dx1x2y1y2 = Self::CURVE_D * x1x2 * y1y2;

                let x3 = (x1y2 + y1x2).div_unsafe(&(Self::Coordinate::ONE + dx1x2y1y2));
                let y3 = (y1y2 - Self::CURVE_A * x1x2).div_unsafe(&(Self::Coordinate::ONE - dx1x2y1y2));

                Self(AffinePoint::new(x3, y3))
            }

            impl core::ops::Neg for $struct_name {
                type Output = Self;

                fn neg(mut self) -> Self::Output {
                    self.0.x.neg_assign();
                    self
                }
            }

            impl core::ops::Neg for &$struct_name {
                type Output = $struct_name;

                fn neg(self) -> Self::Output {
                    self.clone().neg()
                }
            }

            impl From<$struct_name> for AffinePoint<$field> {
                fn from(value: $struct_name) -> Self {
                    value.0
                }
            }

            impl From<AffinePoint<$field>> for $struct_name {
                fn from(value: AffinePoint<$field>) -> Self {
                    Self(value)
                }
            }
        }
    }
}

/// Implements `Group` on `$struct_name` assuming that `$struct_name` implements
/// `TwistedEdwardsPoint`. Assumes that `Neg` is implemented for `&$struct_name`.
#[macro_export]
macro_rules! impl_te_group_ops {
    ($struct_name:ident, $field:ty) => {
        impl Group for $struct_name {
            type SelfRef<'a> = &'a Self;

            const IDENTITY: Self = <Self as TwistedEdwardsPoint>::IDENTITY;

            fn double(&self) -> Self {
                if self.is_identity() {
                    self.clone()
                } else {
                    self.add_impl(self)
                }
            }

            fn double_assign(&mut self) {
                if !self.is_identity() {
                    *self = self.add_impl(self)
                }
            }

            // Note: It was found that implementing `is_identity` in group.rs as a default
            // implementation increases the cycle count by 50% on the ecrecover benchmark. For
            // this reason, we implement it here instead. We hypothesize that this is due to
            // compiler optimizations that are not possible when the `is_identity` function is
            // defined in a different source file.
            #[inline(always)]
            fn is_identity(&self) -> bool {
                self == &<Self as Group>::IDENTITY
            }
        }

        impl core::ops::Add<&$struct_name> for $struct_name {
            type Output = Self;

            fn add(mut self, p2: &$struct_name) -> Self::Output {
                use core::ops::AddAssign;
                self.add_assign(p2);
                self
            }
        }

        impl core::ops::Add for $struct_name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                self.add(&rhs)
            }
        }

        impl core::ops::Add<&$struct_name> for &$struct_name {
            type Output = $struct_name;

            fn add(self, p2: &$struct_name) -> Self::Output {
                if self.is_identity() {
                    p2.clone()
                } else if p2.is_identity() {
                    self.clone()
                } else if self.x() + p2.x() == <$field as openvm_algebra_guest::Field>::ZERO
                    && self.y() == p2.y()
                {
                    <$struct_name as TwistedEdwardsPoint>::IDENTITY
                } else {
                    self.add_impl(p2)
                }
            }
        }

        impl core::ops::AddAssign<&$struct_name> for $struct_name {
            fn add_assign(&mut self, p2: &$struct_name) {
                if self.is_identity() {
                    *self = p2.clone();
                } else if p2.is_identity() {
                    // do nothing
                } else if self.x() + p2.x() == <$field as openvm_algebra_guest::Field>::ZERO
                    && self.y() == p2.y()
                {
                    *self = <$struct_name as TwistedEdwardsPoint>::IDENTITY;
                } else {
                    *self = self.add_impl(p2);
                }
            }
        }

        impl core::ops::AddAssign for $struct_name {
            fn add_assign(&mut self, rhs: Self) {
                self.add_assign(&rhs);
            }
        }

        impl core::ops::Sub<&$struct_name> for $struct_name {
            type Output = Self;

            fn sub(self, rhs: &$struct_name) -> Self::Output {
                core::ops::Sub::sub(&self, rhs)
            }
        }

        impl core::ops::Sub for $struct_name {
            type Output = $struct_name;

            fn sub(self, rhs: Self) -> Self::Output {
                self.sub(&rhs)
            }
        }

        impl core::ops::Sub<&$struct_name> for &$struct_name {
            type Output = $struct_name;

            fn sub(self, p2: &$struct_name) -> Self::Output {
                use core::ops::Add;
                self.add(&-p2)
            }
        }

        impl core::ops::SubAssign<&$struct_name> for $struct_name {
            fn sub_assign(&mut self, p2: &$struct_name) {
                use core::ops::AddAssign;
                self.add_assign(-p2);
            }
        }

        impl core::ops::SubAssign for $struct_name {
            fn sub_assign(&mut self, rhs: Self) {
                self.sub_assign(&rhs);
            }
        }
    };
}

// This is the same as the Weierstrass version, but for Edwards curves we use
// TwistedEdwardsPoint::add_impl instead of WeierstrassPoint::add_ne_nonidentity, etc.
// Unlike the Weierstrass version, we do not require the bases to have prime order, since our
// addition formulas are complete.

// MSM using preprocessed table (windowed method)
// Reference: modified from https://github.com/arkworks-rs/algebra/blob/master/ec/src/scalar_mul/mod.rs

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

impl<'a, C: IntrinsicCurve> CachedMulTable<'a, C>
where
    C::Point: TwistedEdwardsPoint + Group,
    C::Scalar: IntMod,
{
    pub fn new(bases: &'a [C::Point], window_bits: usize) -> Self {
        assert!(window_bits > 0);
        let window_size = 1 << window_bits;
        let table = bases
            .iter()
            .map(|base| {
                if base.is_identity() {
                    vec![<C::Point as Group>::IDENTITY; window_size - 2]
                } else {
                    let mut multiples = Vec::with_capacity(window_size - 2);
                    for _ in 0..window_size - 2 {
                        let multiple = multiples
                            .last()
                            .map(|last| TwistedEdwardsPoint::add_impl(last, base))
                            .unwrap_or_else(|| base.double());
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
            identity: <C::Point as Group>::IDENTITY,
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

        let mut res = <C::Point as Group>::IDENTITY;
        for outer in 0..num_windows {
            if bit_idx == 0 {
                limb_idx -= 1;
                bit_idx = 8 - self.window_bits;
            } else {
                bit_idx -= self.window_bits;
            }

            if outer != 0 {
                for _ in 0..self.window_bits {
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
