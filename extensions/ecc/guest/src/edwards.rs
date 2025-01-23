use core::ops::Mul;

use openvm_algebra_guest::{Field, IntMod};

pub trait TwistedEdwardsPoint: Sized {
    /// The `a` coefficient in the twisted Edwards curve equation `ax^2 + y^2 = 1 + d x^2 y^2`.
    const CURVE_A: Self::Coordinate;
    /// The `d` coefficient in the twisted Edwards curve equation `ax^2 + y^2 = 1 + d x^2 y^2`.
    const CURVE_D: Self::Coordinate;
    const IDENTITY: Self;

    type Coordinate: IntMod + Field;
    const ZERO: Self::Coordinate = <Self::Coordinate as IntMod>::ZERO;
    const ONE: Self::Coordinate = <Self::Coordinate as IntMod>::ONE;

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

    fn from_xy(x: Self::Coordinate, y: Self::Coordinate) -> Option<Self>
    where
        for<'a> &'a Self::Coordinate: Mul<&'a Self::Coordinate, Output = Self::Coordinate>,
    {
        if x == Self::ZERO && y == Self::ONE {
            Some(Self::IDENTITY)
        } else {
            Self::from_xy_nonidentity(x, y)
        }
    }

    fn from_xy_nonidentity(x: Self::Coordinate, y: Self::Coordinate) -> Option<Self>
    where
        for<'a> &'a Self::Coordinate: Mul<&'a Self::Coordinate, Output = Self::Coordinate>,
    {
        let lhs = Self::CURVE_A * &x * &x + &y * &y;
        let rhs = Self::CURVE_D * &x * &x * &y * &y + &Self::ONE;
        if lhs != rhs {
            return None;
        }
        Some(Self::from_xy_unchecked(x, y))
    }
}

/// Macro to generate a newtype wrapper for [AffinePoint](crate::AffinePoint)
/// that implements elliptic curve operations by using the underlying field operations according to the
/// [formulas](https://en.wikipedia.org/wiki/Twisted_Edwards_curve) for twisted Edwards curves.
///
/// The following imports are required:
/// ```rust
/// use core::ops::AddAssign;
///
/// use openvm_algebra_guest::{DivUnsafe, Field};
/// use openvm_ecc_guest::{AffinePoint, Group, edwards::TwistedEdwardsPoint};
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

/// Implements `Group` on `$struct_name` assuming that `$struct_name` implements `TwistedEdwardsPoint`.
/// Assumes that `Neg` is implemented for `&$struct_name`.
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
