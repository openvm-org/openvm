use alloc::vec::Vec;
use core::ops::Mul;

use openvm_algebra_guest::{Field, IntMod};

use super::group::Group;

/// Short Weierstrass curve affine point.
pub trait WeierstrassPoint: Clone + Sized {
    /// The `a` coefficient in the Weierstrass curve equation `y^2 = x^3 + a x + b`.
    const CURVE_A: Self::Coordinate;
    /// The `b` coefficient in the Weierstrass curve equation `y^2 = x^3 + a x + b`.
    const CURVE_B: Self::Coordinate;
    const IDENTITY: Self;

    type Coordinate: Field;

    /// The concatenated `x, y` coordinates of the affine point, where
    /// coordinates are in little endian.
    ///
    /// **Warning**: The memory layout of `Self` is expected to pack
    /// `x` and `y` contiguously with no unallocated space in between.
    fn as_le_bytes(&self) -> &[u8];

    /// Raw constructor without asserting point is on the curve.
    fn from_xy_unchecked(x: Self::Coordinate, y: Self::Coordinate) -> Self;
    fn into_coords(self) -> (Self::Coordinate, Self::Coordinate);
    fn x(&self) -> &Self::Coordinate;
    fn y(&self) -> &Self::Coordinate;
    fn x_mut(&mut self) -> &mut Self::Coordinate;
    fn y_mut(&mut self) -> &mut Self::Coordinate;

    /// Calls any setup required for this curve. The implementation should internally use `OnceBool`
    /// to ensure that setup is only called once.
    fn set_up_once();

    /// Add implementation that handles identity and whether points are equal or not.
    ///
    /// # Safety
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    fn add_assign_impl<const CHECK_SETUP: bool>(&mut self, p2: &Self);

    /// Double implementation that handles identity.
    ///
    /// # Safety
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    fn double_assign_impl<const CHECK_SETUP: bool>(&mut self);

    /// # Safety
    /// - Assumes self != +- p2 and self != identity and p2 != identity.
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    unsafe fn add_ne_nonidentity<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self;
    /// # Safety
    /// - Assumes self != +- p2 and self != identity and p2 != identity.
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    unsafe fn add_ne_assign_nonidentity<const CHECK_SETUP: bool>(&mut self, p2: &Self);
    /// # Safety
    /// - Assumes self != +- p2 and self != identity and p2 != identity.
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    unsafe fn sub_ne_nonidentity<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self;
    /// # Safety
    /// - Assumes self != +- p2 and self != identity and p2 != identity.
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    unsafe fn sub_ne_assign_nonidentity<const CHECK_SETUP: bool>(&mut self, p2: &Self);
    /// # Safety
    /// - Assumes self != identity and 2 * self != identity.
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    unsafe fn double_nonidentity<const CHECK_SETUP: bool>(&self) -> Self;
    /// # Safety
    /// - Assumes self != identity and 2 * self != identity.
    /// - If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    ///   `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    ///   been called already.
    unsafe fn double_assign_nonidentity<const CHECK_SETUP: bool>(&mut self);

    #[inline(always)]
    fn from_xy(x: Self::Coordinate, y: Self::Coordinate) -> Option<Self>
    where
        for<'a> &'a Self::Coordinate: Mul<&'a Self::Coordinate, Output = Self::Coordinate>,
    {
        if x == Self::Coordinate::ZERO && y == Self::Coordinate::ZERO {
            Some(Self::IDENTITY)
        } else {
            Self::from_xy_nonidentity(x, y)
        }
    }

    #[inline(always)]
    fn from_xy_nonidentity(x: Self::Coordinate, y: Self::Coordinate) -> Option<Self>
    where
        for<'a> &'a Self::Coordinate: Mul<&'a Self::Coordinate, Output = Self::Coordinate>,
    {
        let lhs = &y * &y;
        let rhs = &x * &x * &x + &Self::CURVE_A * &x + &Self::CURVE_B;
        if lhs != rhs {
            return None;
        }
        Some(Self::from_xy_unchecked(x, y))
    }
}

pub trait FromCompressed<Coordinate> {
    /// Given `x`-coordinate,
    ///
    /// Decompresses a point from its x-coordinate and a recovery identifier which indicates
    /// the parity of the y-coordinate. Given the x-coordinate, this function attempts to find the
    /// corresponding y-coordinate that satisfies the elliptic curve equation. If successful, it
    /// returns the point as an instance of Self. If the point cannot be decompressed, it returns
    /// None.
    fn decompress(x: Coordinate, rec_id: &u8) -> Option<Self>
    where
        Self: core::marker::Sized;
}

/// A trait for elliptic curves that bridges the openvm types and external types with
/// CurveArithmetic etc. Implement this for external curves with corresponding openvm point and
/// scalar types.
pub trait IntrinsicCurve {
    type Scalar: Clone;
    type Point: Clone;

    /// Multi-scalar multiplication.
    /// The implementation may be specialized to use properties of the curve
    /// (e.g., if the curve order is prime).
    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point;
}

// MSM using preprocessed table (windowed method)
// Reference: modified from https://github.com/arkworks-rs/algebra/blob/master/ec/src/scalar_mul/mod.rs
//
// We specialize to Weierstrass curves and further make optimizations for when the curve order is
// prime.

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
    C::Point: WeierstrassPoint + Group,
    C::Scalar: IntMod,
{
    /// Constructor when each element of `bases` has prime torsion or is identity.
    ///
    /// Assumes that `window_bits` is less than (number of bits - 1) of the order of
    /// subgroup generated by each non-identity `base`.
    #[inline]
    pub fn new_with_prime_order(bases: &'a [C::Point], window_bits: usize) -> Self {
        C::Point::set_up_once();
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
                        // Because the order of `base` is prime, we are guaranteed that
                        // j * base != identity,
                        // j * base != +- base for j > 1,
                        // j * base + base != identity
                        let multiple = multiples
                            .last()
                            .map(|last| unsafe {
                                WeierstrassPoint::add_ne_nonidentity::<false>(last, base)
                            })
                            .unwrap_or_else(|| unsafe { base.double_nonidentity::<false>() });
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

    #[inline(always)]
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
    #[inline]
    pub fn windowed_mul(&self, scalars: &[C::Scalar]) -> C::Point {
        C::Point::set_up_once();
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
                    // Note: this handles identity
                    // setup has been called above
                    res.double_assign_impl::<false>();
                }
            }
            for (base_idx, scalar) in scalars.iter().enumerate() {
                let scalar = (scalar.as_le_bytes()[limb_idx] >> bit_idx) & mask;
                let summand = self.get_multiple(base_idx, scalar as usize);
                // handles identity
                // setup has been called above
                res.add_assign_impl::<false>(summand);
            }
        }
        res
    }
}

/// Macro to generate a newtype wrapper for [AffinePoint](crate::AffinePoint)
/// that implements elliptic curve operations by using the underlying field operations according to
/// the [formulas](https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html) for short Weierstrass curves.
///
/// The following imports are required:
/// ```rust
/// use core::ops::AddAssign;
///
/// use openvm_algebra_guest::{DivUnsafe, Field};
/// use openvm_ecc_guest::{weierstrass::WeierstrassPoint, AffinePoint, Group};
/// ```
#[macro_export]
macro_rules! impl_sw_affine {
    // Assumes `a = 0` in curve equation. `$three` should be a constant expression for `3` of type
    // `$field`.
    ($struct_name:ident, $field:ty, $three:expr, $b:expr) => {
        /// A newtype wrapper for [AffinePoint] that implements elliptic curve operations
        /// by using the underlying field operations according to the [formulas](https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html) for short Weierstrass curves.
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $struct_name(AffinePoint<$field>);

        impl $struct_name {
            pub const fn new(x: $field, y: $field) -> Self {
                Self(AffinePoint::new(x, y))
            }
        }

        impl WeierstrassPoint for $struct_name {
            const CURVE_A: $field = <$field>::ZERO;
            const CURVE_B: $field = $b;
            const IDENTITY: Self = Self(AffinePoint::new(<$field>::ZERO, <$field>::ZERO));

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

            fn set_up_once() {
                // There are no special opcodes for curve operations in this case, so no additional
                // setup is required.
                //
                // Since the `Self::Coordinate` is abstract, any set up required by the field is not
                // handled here.
            }

            fn add_assign_impl<const CHECK_SETUP: bool>(&mut self, p2: &Self) {
                if self == &<Self as WeierstrassPoint>::IDENTITY {
                    *self = p2.clone();
                } else if p2 == &<Self as WeierstrassPoint>::IDENTITY {
                    // do nothing
                } else if self.x() == p2.x() {
                    if self.y() + p2.y() == <Self::Coordinate as openvm_algebra_guest::Field>::ZERO
                    {
                        *self = <Self as WeierstrassPoint>::IDENTITY;
                    } else {
                        unsafe {
                            self.double_assign_nonidentity::<CHECK_SETUP>();
                        }
                    }
                } else {
                    unsafe {
                        self.add_ne_assign_nonidentity::<CHECK_SETUP>(p2);
                    }
                }
            }

            #[inline(always)]
            fn double_assign_impl<const CHECK_SETUP: bool>(&mut self) {
                if self != &<Self as WeierstrassPoint>::IDENTITY {
                    unsafe {
                        self.double_assign_nonidentity::<CHECK_SETUP>();
                    }
                }
            }

            unsafe fn double_nonidentity<const CHECK_SETUP: bool>(&self) -> Self {
                use openvm_algebra_guest::DivUnsafe;
                // lambda = (3*x1^2+a)/(2*y1)
                // assume a = 0
                let lambda = (&THREE * self.x() * self.x()).div_unsafe(self.y() + self.y());
                // x3 = lambda^2-x1-x1
                let x3 = &lambda * &lambda - self.x() - self.x();
                // y3 = lambda * (x1-x3) - y1
                let y3 = lambda * (self.x() - &x3) - self.y();
                Self(AffinePoint::new(x3, y3))
            }

            #[inline(always)]
            unsafe fn double_assign_nonidentity<const CHECK_SETUP: bool>(&mut self) {
                *self = self.double_nonidentity::<CHECK_SETUP>();
            }

            unsafe fn add_ne_nonidentity<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self {
                use openvm_algebra_guest::DivUnsafe;
                // lambda = (y2-y1)/(x2-x1)
                // x3 = lambda^2-x1-x2
                // y3 = lambda*(x1-x3)-y1
                let lambda = (p2.y() - self.y()).div_unsafe(p2.x() - self.x());
                let x3 = &lambda * &lambda - self.x() - p2.x();
                let y3 = lambda * (self.x() - &x3) - self.y();
                Self(AffinePoint::new(x3, y3))
            }

            #[inline(always)]
            unsafe fn add_ne_assign_nonidentity<const CHECK_SETUP: bool>(&mut self, p2: &Self) {
                *self = self.add_ne_nonidentity::<CHECK_SETUP>(p2);
            }

            unsafe fn sub_ne_nonidentity<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self {
                use openvm_algebra_guest::DivUnsafe;
                // lambda = (y2+y1)/(x1-x2)
                // x3 = lambda^2-x1-x2
                // y3 = lambda*(x1-x3)-y1
                let lambda = (p2.y() + self.y()).div_unsafe(self.x() - p2.x());
                let x3 = &lambda * &lambda - self.x() - p2.x();
                let y3 = lambda * (self.x() - &x3) - self.y();
                Self(AffinePoint::new(x3, y3))
            }

            #[inline(always)]
            unsafe fn sub_ne_assign_nonidentity<const CHECK_SETUP: bool>(&mut self, p2: &Self) {
                *self = self.sub_ne_nonidentity::<CHECK_SETUP>(p2);
            }
        }

        impl core::ops::Neg for $struct_name {
            type Output = Self;

            #[inline(always)]
            fn neg(mut self) -> Self::Output {
                self.0.y.neg_assign();
                self
            }
        }

        impl core::ops::Neg for &$struct_name {
            type Output = $struct_name;

            #[inline(always)]
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
    };
}

/// Implements `Group` on `$struct_name` assuming that `$struct_name` implements `WeierstrassPoint`.
/// Assumes that `Neg` is implemented for `&$struct_name`.
#[macro_export]
macro_rules! impl_sw_group_ops {
    ($struct_name:ident, $field:ty) => {
        impl Group for $struct_name {
            type SelfRef<'a> = &'a Self;

            const IDENTITY: Self = <Self as WeierstrassPoint>::IDENTITY;

            #[inline(always)]
            fn double(&self) -> Self {
                if self.is_identity() {
                    self.clone()
                } else {
                    unsafe { self.double_nonidentity::<true>() }
                }
            }

            #[inline(always)]
            fn double_assign(&mut self) {
                self.double_assign_impl::<true>();
            }

            // This implementation is the same as the default implementation in the `Group` trait,
            // but it was found that overriding the default implementation reduced the cycle count
            // by 50% on the ecrecover benchmark.
            // We hypothesize that this is due to compiler optimizations that are not possible when
            // the `is_identity` function is defined in a different source file.
            #[inline(always)]
            fn is_identity(&self) -> bool {
                self == &<Self as Group>::IDENTITY
            }
        }

        impl core::ops::Add<&$struct_name> for $struct_name {
            type Output = Self;

            #[inline(always)]
            fn add(mut self, p2: &$struct_name) -> Self::Output {
                use core::ops::AddAssign;
                self.add_assign(p2);
                self
            }
        }

        impl core::ops::Add for $struct_name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self::Output {
                self.add(&rhs)
            }
        }

        impl core::ops::Add<&$struct_name> for &$struct_name {
            type Output = $struct_name;

            #[inline(always)]
            fn add(self, p2: &$struct_name) -> Self::Output {
                if self.is_identity() {
                    p2.clone()
                } else if p2.is_identity() {
                    self.clone()
                } else if WeierstrassPoint::x(self) == WeierstrassPoint::x(p2) {
                    if self.y() + p2.y() == <$field as openvm_algebra_guest::Field>::ZERO {
                        <$struct_name as WeierstrassPoint>::IDENTITY
                    } else {
                        unsafe { self.double_nonidentity::<true>() }
                    }
                } else {
                    unsafe { self.add_ne_nonidentity::<true>(p2) }
                }
            }
        }

        impl core::ops::AddAssign<&$struct_name> for $struct_name {
            #[inline(always)]
            fn add_assign(&mut self, p2: &$struct_name) {
                self.add_assign_impl::<true>(p2);
            }
        }

        impl core::ops::AddAssign for $struct_name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.add_assign(&rhs);
            }
        }

        impl core::ops::Sub<&$struct_name> for $struct_name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: &$struct_name) -> Self::Output {
                core::ops::Sub::sub(&self, rhs)
            }
        }

        impl core::ops::Sub for $struct_name {
            type Output = $struct_name;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self::Output {
                self.sub(&rhs)
            }
        }

        impl core::ops::Sub<&$struct_name> for &$struct_name {
            type Output = $struct_name;

            #[inline(always)]
            fn sub(self, p2: &$struct_name) -> Self::Output {
                if p2.is_identity() {
                    self.clone()
                } else if self.is_identity() {
                    core::ops::Neg::neg(p2)
                } else if WeierstrassPoint::x(self) == WeierstrassPoint::x(p2) {
                    if self.y() == p2.y() {
                        <$struct_name as WeierstrassPoint>::IDENTITY
                    } else {
                        unsafe { self.double_nonidentity::<true>() }
                    }
                } else {
                    unsafe { self.sub_ne_nonidentity::<true>(p2) }
                }
            }
        }

        impl core::ops::SubAssign<&$struct_name> for $struct_name {
            #[inline(always)]
            fn sub_assign(&mut self, p2: &$struct_name) {
                if p2.is_identity() {
                    // do nothing
                } else if self.is_identity() {
                    *self = core::ops::Neg::neg(p2);
                } else if WeierstrassPoint::x(self) == WeierstrassPoint::x(p2) {
                    if self.y() == p2.y() {
                        *self = <$struct_name as WeierstrassPoint>::IDENTITY
                    } else {
                        unsafe {
                            self.double_assign_nonidentity::<true>();
                        }
                    }
                } else {
                    unsafe {
                        self.sub_ne_assign_nonidentity::<true>(p2);
                    }
                }
            }
        }

        impl core::ops::SubAssign for $struct_name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.sub_assign(&rhs);
            }
        }
    };
}
