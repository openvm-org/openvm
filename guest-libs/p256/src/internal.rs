use core::ops::{Add, Neg};

use hex_literal::hex;
use openvm_algebra_guest::{IntMod, Reduce};
use openvm_algebra_moduli_macros::moduli_declare;
use openvm_ecc_guest::{
    weierstrass::{IntrinsicCurve, ScalarMul, WeierstrassPoint},
    CyclicGroup, Group,
};
use openvm_ecc_sw_macros::sw_declare;

use crate::NistP256;

// --- Define the OpenVM modular arithmetic and ecc types ---

moduli_declare! {
    P256Coord { modulus = "0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff" },
    P256Scalar { modulus = "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551" },
}

// from_const_bytes is little endian
pub const CURVE_A: P256Coord = P256Coord::from_const_bytes(hex!(
    "fcffffffffffffffffffffff00000000000000000000000001000000ffffffff"
));
pub const CURVE_B: P256Coord = P256Coord::from_const_bytes(hex!(
    "4b60d2273e3cce3bf6b053ccb0061d65bc86987655bdebb3e7933aaad835c65a"
));

sw_declare! {
    P256Point { mod_type = P256Coord, a = CURVE_A, b = CURVE_B },
}

// --- Implement internal traits ---

impl CyclicGroup for P256Point {
    // The constants are taken from: https://neuromancer.sk/std/secg/secp256r1
    const GENERATOR: Self = P256Point {
        // from_const_bytes takes a little endian byte string
        x: P256Coord::from_const_bytes(hex!(
            "96c298d84539a1f4a033eb2d817d0377f240a463e5e6bcf847422ce1f2d1176b"
        )),
        y: P256Coord::from_const_bytes(hex!(
            "f551bf376840b6cbce5e316b5733ce2b169e0f7c4aebe78e9b7f1afee242e34f"
        )),
    };
    const NEG_GENERATOR: Self = P256Point {
        x: P256Coord::from_const_bytes(hex!(
            "96c298d84539a1f4a033eb2d817d0377f240a463e5e6bcf847422ce1f2d1176b"
        )),
        y: P256Coord::from_const_bytes(hex!(
            "0aae40c897bf493431a1ce94a9cc31d4e961f083b51418716580e5011cbd1cb0"
        )),
    };
}

impl IntrinsicCurve for NistP256 {
    type Scalar = P256Scalar;
    type Point = P256Point;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
    where
        for<'a> &'a Self::Point: Add<&'a Self::Point, Output = Self::Point>,
    {
        assert_eq!(coeffs.len(), bases.len());

        let mut acc = <Self::Point as Group>::IDENTITY;
        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            acc += base.mul_scalar(coeff);
        }
        acc
    }
}

// --- Implement helpful methods mimicking the structs in p256 ---

impl P256Point {
    pub fn x_be_bytes(&self) -> [u8; 32] {
        <Self as WeierstrassPoint>::x(self).to_be_bytes()
    }

    pub fn y_be_bytes(&self) -> [u8; 32] {
        <Self as WeierstrassPoint>::y(self).to_be_bytes()
    }

    /// Returns `scalar * self`, for any scalar representation and any point.
    ///
    /// [`P256Point::mul_scalar_le_unchecked`] requires a non-identity base point and a scalar below
    /// the group order; this discharges both preconditions, so it is total.
    ///
    /// `P256Scalar` admits unreduced representations, since `from_le_bytes_unchecked` and
    /// `from_be_bytes_unchecked` do not reduce. P-256 has cofactor 1, so every point on the curve
    /// has order dividing the group order and reducing the scalar leaves the product unchanged.
    pub fn mul_scalar(&self, scalar: &P256Scalar) -> Self {
        if self.is_identity() {
            return <Self as Group>::IDENTITY;
        }
        let mut reduced = P256Scalar::reduce_le_bytes(scalar.as_le_bytes());
        // Zero admits no odd representative: negating it yields `n - 0 = 0`.
        if reduced == P256Scalar::ZERO {
            return <Self as Group>::IDENTITY;
        }

        // The intrinsic expands the scalar into digits drawn from `{+1, -1}`, whose sum is odd for
        // every choice of signs; an even scalar therefore has no digit assignment and would produce
        // an unprovable trace. Substituting `n - k` restores oddness, `n` itself being odd, and
        // preserves the order bound. The substitution is exact: `(n - k) * P = -(k * P)`, so
        // negating the result recovers the product.
        let odd = reduced.as_le_bytes()[0] & 1 == 1;
        if !odd {
            reduced.neg_assign();
        }
        let bytes: [u8; 32] = reduced.as_le_bytes().try_into().unwrap();
        // SAFETY: `self` is not the identity, and `reduced` is odd and below the group order.
        let product = unsafe { self.mul_scalar_le_unchecked(&bytes) };
        if odd {
            product
        } else {
            -product
        }
    }
}

impl ScalarMul<P256Scalar> for P256Point {
    fn mul_scalar(&self, scalar: &P256Scalar) -> Self {
        P256Point::mul_scalar(self, scalar)
    }
}
