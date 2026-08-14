use core::ops::Neg;

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

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point {
        openvm_ecc_guest::msm_via_ec_mul(coeffs, bases)
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

    pub fn mul_scalar(&self, scalar: &P256Scalar) -> Self {
        if self.is_identity() {
            return <Self as Group>::IDENTITY;
        }
        let mut reduced = P256Scalar::reduce_le_bytes(scalar.as_le_bytes());
        if reduced == P256Scalar::ZERO {
            return <Self as Group>::IDENTITY;
        }

        // The intrinsic needs an odd scalar below n; substitute the odd n - k and negate the
        // product, since (n - k) * P = -(k * P).
        let odd = reduced.as_le_bytes()[0] & 1 == 1;
        if !odd {
            reduced.neg_assign();
        }
        let bytes: [u8; 32] = reduced.as_le_bytes().try_into().unwrap();
        // SAFETY: `self` is not the identity, and `reduced` is odd and below the group order.
        let product = unsafe { self.mul_scalar_le_unchecked::<true>(&bytes) };
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
