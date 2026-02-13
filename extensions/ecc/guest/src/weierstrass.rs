use alloc::vec::Vec;
use core::ops::Mul;

use openvm_algebra_guest::{Field, IntMod};

use super::group::Group;

/// Short Weierstrass curve point in projective coordinates (X, Y, Z).
/// The affine point is (X/Z, Y/Z). The identity is represented as (0, 1, 0).
pub trait WeierstrassPoint: Clone + Sized {
    /// The `a` coefficient in the Weierstrass curve equation `y^2 = x^3 + a x + b`.
    const CURVE_A: Self::Coordinate;
    /// The `b` coefficient in the Weierstrass curve equation `y^2 = x^3 + a x + b`.
    const CURVE_B: Self::Coordinate;
    const IDENTITY: Self;

    type Coordinate: Field;

    /// The concatenated `x, y, z` coordinates of the projective point, where
    /// coordinates are in little endian.
    ///
    /// **Warning**: The memory layout of `Self` is expected to pack
    /// `x`, `y`, and `z` contiguously with no unallocated space in between.
    fn as_le_bytes(&self) -> &[u8];

    /// Raw constructor without asserting point is on the curve. Sets z = 1.
    fn from_xy_unchecked(x: Self::Coordinate, y: Self::Coordinate) -> Self;
    /// Raw constructor from projective coordinates.
    fn from_xyz_unchecked(x: Self::Coordinate, y: Self::Coordinate, z: Self::Coordinate) -> Self;
    fn into_coords(self) -> (Self::Coordinate, Self::Coordinate, Self::Coordinate);
    fn x(&self) -> &Self::Coordinate;
    fn y(&self) -> &Self::Coordinate;
    fn z(&self) -> &Self::Coordinate;
    fn x_mut(&mut self) -> &mut Self::Coordinate;
    fn y_mut(&mut self) -> &mut Self::Coordinate;
    fn z_mut(&mut self) -> &mut Self::Coordinate;

    /// Calls any setup required for this curve. The implementation should internally use `OnceBool`
    /// to ensure that setup is only called once.
    fn set_up_once();

    /// Complete projective addition formula.
    ///
    /// If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    /// `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    /// been called already.
    fn add_impl<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self;

    /// Complete projective doubling formula.
    ///
    /// If `CHECK_SETUP` is true, checks if setup has been called for this curve and if not, calls
    /// `Self::set_up_once()`. Only set `CHECK_SETUP` to `false` if you are sure that setup has
    /// been called already.
    fn double_impl<const CHECK_SETUP: bool>(&self) -> Self;

    /// Normalize to affine coordinates: (X/Z, Y/Z, 1). Returns identity for z=0.
    fn normalize(&self) -> Self;

    /// Check if this point is the identity (z == 0).
    fn is_identity(&self) -> bool;

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

/// Macro to generate a projective point type for short Weierstrass curves.
/// Implements elliptic curve operations using complete projective formulas
/// from ePrint 2015/1060 (no DivUnsafe needed for add/double).
///
/// The following imports are required:
/// ```rust
/// use openvm_algebra_guest::Field;
/// use openvm_ecc_guest::weierstrass::WeierstrassPoint;
/// ```
#[macro_export]
macro_rules! impl_sw_proj {
    // Assumes `a = 0` in curve equation `y^2 = x^3 + a*x + b`.
    ($struct_name:ident, $field:ty, $b:expr) => {
        /// A projective point on a short Weierstrass curve, implementing elliptic curve operations
        /// using complete formulas from [ePrint 2015/1060](https://eprint.iacr.org/2015/1060).
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
        #[repr(C)]
        pub struct $struct_name {
            pub x: $field,
            pub y: $field,
            pub z: $field,
        }

        impl $struct_name {
            pub const fn new(x: $field, y: $field, z: $field) -> Self {
                Self { x, y, z }
            }

            /// Complete projective addition for a=0 curves.
            /// Algorithm 7 from ePrint 2015/1060.
            #[inline(always)]
            fn add_a0(
                x1: &$field,
                y1: &$field,
                z1: &$field,
                x2: &$field,
                y2: &$field,
                z2: &$field,
                b3: &$field,
            ) -> Self {
                let t0 = x1 * x2;
                let t1 = y1 * y2;
                let t2 = z1 * z2;
                let t3 = &(x1 + y1) * &(x2 + y2) - &t0 - &t1;
                let t4 = &(y1 + z1) * &(y2 + z2) - &t1 - &t2;
                let y3_temp = &(x1 + z1) * &(x2 + z2) - &t0 - &t2;
                let x3_coeff = &t0 + &t0 + &t0; // 3*t0
                let t2_b3 = b3 * &t2;
                let z3_temp = &t1 + &t2_b3;
                let t1_sub = &t1 - &t2_b3;
                let y3_b3 = b3 * &y3_temp;
                let x3_out = &t3 * &t1_sub - &t4 * &y3_b3;
                let y3_out = &t1_sub * &z3_temp + &y3_b3 * &x3_coeff;
                let z3_out = &z3_temp * &t4 + &x3_coeff * &t3;
                Self {
                    x: x3_out,
                    y: y3_out,
                    z: z3_out,
                }
            }

            /// Complete projective doubling for a=0 curves.
            /// Algorithm 9 from ePrint 2015/1060.
            #[inline(always)]
            fn double_a0(x1: &$field, y1: &$field, z1: &$field, b3: &$field) -> Self {
                let t0 = y1 * y1;
                let t0_2 = &t0 + &t0;
                let t0_4 = &t0_2 + &t0_2;
                let z3_8t0 = &t0_4 + &t0_4; // 8*t0
                let t1 = y1 * z1;
                let z1_sq = z1 * z1;
                let t2 = b3 * &z1_sq;
                let x3_temp = &t2 * &z3_8t0;
                let y3 = &t0 + &t2;
                let z3 = &t1 * &z3_8t0;
                let t2_2 = &t2 + &t2;
                let t2_3 = &t2_2 + &t2;
                let t0 = &t0 - &t2_3;
                let y3_part = &t0 * &y3;
                let y3 = &y3_part + &x3_temp;
                let t1 = x1 * y1;
                let x3_part = &t0 * &t1;
                let x3 = &x3_part + &x3_part; // 2 * t0 * t1
                Self {
                    x: x3,
                    y: y3,
                    z: z3,
                }
            }
        }

        impl WeierstrassPoint for $struct_name {
            const CURVE_A: $field = <$field>::ZERO;
            const CURVE_B: $field = $b;
            const IDENTITY: Self = Self {
                x: <$field>::ZERO,
                y: <$field>::ONE,
                z: <$field>::ZERO,
            };

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
                Self {
                    x,
                    y,
                    z: <$field>::ONE,
                }
            }
            fn from_xyz_unchecked(
                x: Self::Coordinate,
                y: Self::Coordinate,
                z: Self::Coordinate,
            ) -> Self {
                Self { x, y, z }
            }
            fn into_coords(self) -> (Self::Coordinate, Self::Coordinate, Self::Coordinate) {
                (self.x, self.y, self.z)
            }
            fn x(&self) -> &Self::Coordinate {
                &self.x
            }
            fn y(&self) -> &Self::Coordinate {
                &self.y
            }
            fn z(&self) -> &Self::Coordinate {
                &self.z
            }
            fn x_mut(&mut self) -> &mut Self::Coordinate {
                &mut self.x
            }
            fn y_mut(&mut self) -> &mut Self::Coordinate {
                &mut self.y
            }
            fn z_mut(&mut self) -> &mut Self::Coordinate {
                &mut self.z
            }

            fn set_up_once() {
                // No special opcodes for curve operations in this case.
            }

            fn add_impl<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self {
                let b3 = &(&<Self as WeierstrassPoint>::CURVE_B
                    + &<Self as WeierstrassPoint>::CURVE_B)
                    + &<Self as WeierstrassPoint>::CURVE_B;
                Self::add_a0(&self.x, &self.y, &self.z, &p2.x, &p2.y, &p2.z, &b3)
            }

            fn double_impl<const CHECK_SETUP: bool>(&self) -> Self {
                let b3 = &(&<Self as WeierstrassPoint>::CURVE_B
                    + &<Self as WeierstrassPoint>::CURVE_B)
                    + &<Self as WeierstrassPoint>::CURVE_B;
                Self::double_a0(&self.x, &self.y, &self.z, &b3)
            }

            fn normalize(&self) -> Self {
                use openvm_algebra_guest::DivUnsafe;
                if self.z == <$field>::ZERO {
                    <Self as WeierstrassPoint>::IDENTITY
                } else {
                    let x = (&self.x).div_unsafe(&self.z);
                    let y = (&self.y).div_unsafe(&self.z);
                    Self {
                        x,
                        y,
                        z: <$field>::ONE,
                    }
                }
            }

            fn is_identity(&self) -> bool {
                self.z == <$field>::ZERO
            }
        }

        impl core::ops::Neg for $struct_name {
            type Output = Self;

            #[inline(always)]
            fn neg(mut self) -> Self::Output {
                self.y.neg_assign();
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
    };
}

/// Implements `Group` on `$struct_name` assuming that `$struct_name` implements `WeierstrassPoint`.
/// Assumes that `Neg` is implemented for `&$struct_name`.
/// Complete projective formulas handle all edge cases, so no branching is needed.
#[macro_export]
macro_rules! impl_sw_group_ops {
    ($struct_name:ident, $field:ty) => {
        impl Group for $struct_name {
            type SelfRef<'a> = &'a Self;

            const IDENTITY: Self = <Self as WeierstrassPoint>::IDENTITY;

            #[inline(always)]
            fn double(&self) -> Self {
                self.double_impl::<true>()
            }

            #[inline(always)]
            fn double_assign(&mut self) {
                *self = self.double_impl::<true>();
            }

            #[inline(always)]
            fn is_identity(&self) -> bool {
                <Self as WeierstrassPoint>::is_identity(self)
            }
        }

        impl core::ops::Add<&$struct_name> for $struct_name {
            type Output = Self;

            #[inline(always)]
            fn add(self, p2: &$struct_name) -> Self::Output {
                self.add_impl::<true>(p2)
            }
        }

        impl core::ops::Add for $struct_name {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self::Output {
                self.add_impl::<true>(&rhs)
            }
        }

        impl core::ops::Add<&$struct_name> for &$struct_name {
            type Output = $struct_name;

            #[inline(always)]
            fn add(self, p2: &$struct_name) -> Self::Output {
                self.add_impl::<true>(p2)
            }
        }

        impl core::ops::AddAssign<&$struct_name> for $struct_name {
            #[inline(always)]
            fn add_assign(&mut self, p2: &$struct_name) {
                *self = self.add_impl::<true>(p2);
            }
        }

        impl core::ops::AddAssign for $struct_name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = self.add_impl::<true>(&rhs);
            }
        }

        impl core::ops::Sub<&$struct_name> for $struct_name {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: &$struct_name) -> Self::Output {
                self.add_impl::<true>(&core::ops::Neg::neg(rhs))
            }
        }

        impl core::ops::Sub for $struct_name {
            type Output = $struct_name;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self::Output {
                self.add_impl::<true>(&core::ops::Neg::neg(rhs))
            }
        }

        impl core::ops::Sub<&$struct_name> for &$struct_name {
            type Output = $struct_name;

            #[inline(always)]
            fn sub(self, p2: &$struct_name) -> Self::Output {
                self.add_impl::<true>(&core::ops::Neg::neg(p2))
            }
        }

        impl core::ops::SubAssign<&$struct_name> for $struct_name {
            #[inline(always)]
            fn sub_assign(&mut self, p2: &$struct_name) {
                *self = self.add_impl::<true>(&core::ops::Neg::neg(p2));
            }
        }

        impl core::ops::SubAssign for $struct_name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.add_impl::<true>(&core::ops::Neg::neg(rhs));
            }
        }
    };
}
