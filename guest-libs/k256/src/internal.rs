use core::ops::{Add, Neg};

use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_algebra_moduli_macros::moduli_declare;
use openvm_ecc_guest::{
    weierstrass::{CachedMulTable, IntrinsicCurve, WeierstrassPoint},
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

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
    where
        for<'a> &'a Self::Point: Add<&'a Self::Point, Output = Self::Point>,
    {
        // heuristic
        if coeffs.len() < 25 {
            let table = CachedMulTable::<Self>::new_with_prime_order(bases, 4);
            table.windowed_mul(coeffs)
        } else {
            openvm_ecc_guest::msm(coeffs, bases)
        }
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
}

// Host-side coverage for the ladder that `sw_declare!` generates for non-openvm targets. Guest
// programs are run on the host through that path, so it has to agree with the windowed method the
// rest of the guest library uses.
#[cfg(all(test, not(any(openvm_intrinsics, target_os = "openvm"))))]
mod tests {
    use openvm_algebra_guest::IntMod;
    use openvm_ecc_guest::{weierstrass::IntrinsicCurve, CyclicGroup};

    use super::{Secp256k1, Secp256k1Point, Secp256k1Scalar};

    #[test]
    fn mul_scalar_matches_msm() {
        for k in [1u64, 2, 3, 255, 0x1234_5678, u32::MAX as u64] {
            let scalar = Secp256k1Scalar::from_u64(k);
            let bytes: [u8; 32] = scalar.as_le_bytes().try_into().unwrap();

            let via_ladder = unsafe { Secp256k1Point::GENERATOR.mul_scalar_le_unchecked(&bytes) };
            let via_msm = Secp256k1::msm(&[scalar], &[Secp256k1Point::GENERATOR]);

            assert_eq!(via_ladder, via_msm, "k = {k}");
        }
    }
}
