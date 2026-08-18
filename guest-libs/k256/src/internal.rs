use core::ops::Neg;

use hex_literal::hex;
use openvm_algebra_guest::{IntMod, Reduce};
use openvm_algebra_moduli_macros::moduli_declare;
use openvm_ecc_guest::{
    weierstrass::{IntrinsicCurve, ScalarMul, WeierstrassPoint},
    CyclicGroup, Group,
};
use openvm_ecc_sw_macros::sw_declare;

use crate::Secp256k1;

// --- Define the OpenVM modular arithmetic and ecc types ---

const CURVE_B: Secp256k1Coord = Secp256k1Coord::from_const_bytes(seven_le());
pub const fn seven_le() -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[0] = 7;
    buf
}

moduli_declare! {
    Secp256k1Coord { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F" },
    Secp256k1Scalar { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141" },
}

sw_declare! {
    Secp256k1Point { mod_type = Secp256k1Coord, b = CURVE_B },
}

// --- Implement internal traits ---

impl CyclicGroup for Secp256k1Point {
    // The constants are taken from: https://en.bitcoin.it/wiki/Secp256k1
    const GENERATOR: Self = Secp256k1Point {
        // from_const_bytes takes a little endian byte string
        x: Secp256k1Coord::from_const_bytes(hex!(
            "9817F8165B81F259D928CE2DDBFC9B02070B87CE9562A055ACBBDCF97E66BE79"
        )),
        y: Secp256k1Coord::from_const_bytes(hex!(
            "B8D410FB8FD0479C195485A648B417FDA808110EFCFBA45D65C4A32677DA3A48"
        )),
    };
    const NEG_GENERATOR: Self = Secp256k1Point {
        x: Secp256k1Coord::from_const_bytes(hex!(
            "9817F8165B81F259D928CE2DDBFC9B02070B87CE9562A055ACBBDCF97E66BE79"
        )),
        y: Secp256k1Coord::from_const_bytes(hex!(
            "7727EF046F2FB863E6AB7A59B74BE80257F7EEF103045BA29A3B5CD98825C5B7"
        )),
    };
}

impl IntrinsicCurve for Secp256k1 {
    type Scalar = Secp256k1Scalar;
    type Point = Secp256k1Point;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point {
        openvm_ecc_guest::msm_via_ec_mul(coeffs, bases)
    }
}

// --- Implement helpful methods mimicking the structs in k256 ---

impl Secp256k1Point {
    pub fn x_be_bytes(&self) -> [u8; 32] {
        <Self as WeierstrassPoint>::x(self).to_be_bytes()
    }

    pub fn y_be_bytes(&self) -> [u8; 32] {
        <Self as WeierstrassPoint>::y(self).to_be_bytes()
    }

    pub fn mul_scalar(&self, scalar: &Secp256k1Scalar) -> Self {
        if self.is_identity() {
            return <Self as Group>::IDENTITY;
        }
        let mut reduced = Secp256k1Scalar::reduce_le_bytes(scalar.as_le_bytes());
        if reduced == Secp256k1Scalar::ZERO {
            return <Self as Group>::IDENTITY;
        }

        // The intrinsic needs an odd scalar below n; substitute the odd n - k and negate the
        // product, since (n - k) * P = -(k * P).
        let odd = reduced.as_le_bytes()[0] & 1 == 1;
        if !odd {
            reduced.neg_assign();
        }
        let bytes: [u8; 32] = reduced.as_le_bytes().try_into().unwrap();
        // SAFETY: Every valid point is in this prime-order group. Its order is 1 modulo 4.
        // `reduced` is odd, nonzero, and less than the order. EC_MUL setup runs on first use.
        let product = unsafe { self.mul_scalar_le_unchecked::<true>(&bytes) };
        if odd {
            product
        } else {
            -product
        }
    }
}

impl ScalarMul<Secp256k1Scalar> for Secp256k1Point {
    fn mul_scalar(&self, scalar: &Secp256k1Scalar) -> Self {
        Secp256k1Point::mul_scalar(self, scalar)
    }
}

#[cfg(all(test, not(any(openvm_intrinsics, target_os = "openvm"))))]
mod tests {
    use hex_literal::hex;
    use openvm_algebra_guest::IntMod;
    use openvm_ecc_guest::{weierstrass::CachedMulTable, CyclicGroup, Group};

    use super::{Secp256k1, Secp256k1Point, Secp256k1Scalar};

    fn windowed_reference(scalar: Secp256k1Scalar) -> Secp256k1Point {
        let base = [Secp256k1Point::GENERATOR];
        let table = CachedMulTable::<Secp256k1>::new_with_prime_order(&base, 4);
        table.windowed_mul(&[scalar])
    }

    #[test]
    fn mul_scalar_matches_windowed() {
        for k in [1u64, 3, 5, 255, 0x1234_5679, u32::MAX as u64] {
            let scalar = Secp256k1Scalar::from_u64(k);
            let bytes: [u8; 32] = scalar.as_le_bytes().try_into().unwrap();

            let via_ladder =
                unsafe { Secp256k1Point::GENERATOR.mul_scalar_le_unchecked::<true>(&bytes) };

            assert_eq!(via_ladder, windowed_reference(scalar), "k = {k}");
        }
    }

    #[test]
    fn mul_scalar_reduces_the_scalar() {
        // n + 5
        let unreduced = Secp256k1Scalar::from_be_bytes_unchecked(&hex!(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364146"
        ));
        assert!(!unreduced.is_reduced());

        let expected = windowed_reference(Secp256k1Scalar::from_u64(5));
        assert_eq!(Secp256k1Point::GENERATOR.mul_scalar(&unreduced), expected);
    }

    #[test]
    fn mul_scalar_handles_the_identity() {
        let identity = <Secp256k1Point as Group>::IDENTITY;
        for k in [0u64, 1, 7] {
            let scalar = Secp256k1Scalar::from_u64(k);
            assert!(identity.mul_scalar(&scalar).is_identity(), "k = {k}");
        }

        let zero = Secp256k1Scalar::ZERO;
        assert!(Secp256k1Point::GENERATOR.mul_scalar(&zero).is_identity());
    }
}
