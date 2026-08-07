use core::ops::{Add, Neg};

use hex_literal::hex;
use openvm_algebra_guest::{IntMod, Reduce};
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
        if let ([coeff], [base]) = (coeffs, bases) {
            return base.mul_scalar(coeff);
        }

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

    /// Returns `scalar * self`, for any scalar representation and any point.
    ///
    /// [`Secp256k1Point::mul_scalar_le_unchecked`] requires a non-identity base point and a scalar
    /// that is odd and below the group order; this discharges all three preconditions, so it is
    /// total.
    ///
    /// `Secp256k1Scalar` admits unreduced representations, since `from_le_bytes_unchecked` and
    /// `from_be_bytes_unchecked` do not reduce. secp256k1 has cofactor 1, so every point on the
    /// curve has order dividing the group order and reducing the scalar leaves the product
    /// unchanged.
    pub fn mul_scalar(&self, scalar: &Secp256k1Scalar) -> Self {
        if self.is_identity() {
            return <Self as Group>::IDENTITY;
        }
        let mut reduced = Secp256k1Scalar::reduce_le_bytes(scalar.as_le_bytes());
        // Zero admits no odd representative: negating it yields `n - 0 = 0`.
        if reduced == Secp256k1Scalar::ZERO {
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

// Host-side coverage for the ladder that `sw_declare!` generates for non-openvm targets. Guest
// programs are run on the host through that path, so it has to agree with the windowed method the
// rest of the guest library uses.
#[cfg(all(test, not(any(openvm_intrinsics, target_os = "openvm"))))]
mod tests {
    use hex_literal::hex;
    use openvm_algebra_guest::IntMod;
    use openvm_ecc_guest::{weierstrass::CachedMulTable, CyclicGroup, Group};

    use super::{Secp256k1, Secp256k1Point, Secp256k1Scalar};

    /// Reference implementation: the windowed table, called directly.
    ///
    /// `IntrinsicCurve::msm` cannot serve as the reference any more, because it now routes a
    /// single pair to `mul_scalar` and would compare the ladder against itself.
    fn windowed_reference(scalar: Secp256k1Scalar) -> Secp256k1Point {
        let base = [Secp256k1Point::GENERATOR];
        let table = CachedMulTable::<Secp256k1>::new_with_prime_order(&base, 4);
        table.windowed_mul(&[scalar])
    }

    #[test]
    fn mul_scalar_matches_windowed() {
        for k in [1u64, 2, 3, 255, 0x1234_5678, u32::MAX as u64] {
            let scalar = Secp256k1Scalar::from_u64(k);
            let bytes: [u8; 32] = scalar.as_le_bytes().try_into().unwrap();

            let via_ladder = unsafe { Secp256k1Point::GENERATOR.mul_scalar_le_unchecked(&bytes) };

            assert_eq!(via_ladder, windowed_reference(scalar), "k = {k}");
        }
    }

    #[test]
    fn mul_scalar_reduces_the_scalar() {
        // `n + 5`, one group order above a small scalar. `Secp256k1Scalar` stores it verbatim, so
        // this is the representation `mul_scalar` has to reduce before reaching the ladder.
        let unreduced = Secp256k1Scalar::from_be_bytes_unchecked(&hex!(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364146"
        ));
        assert!(!unreduced.is_reduced());

        let expected = windowed_reference(Secp256k1Scalar::from_u64(5));
        assert_eq!(Secp256k1Point::GENERATOR.mul_scalar(&unreduced), expected);
    }

    #[test]
    fn mul_scalar_handles_the_identity() {
        // `k * O = O` for every `k`. The ladder itself cannot prove this case, so `mul_scalar`
        // short-circuits it.
        let identity = <Secp256k1Point as Group>::IDENTITY;
        for k in [0u64, 1, 7] {
            let scalar = Secp256k1Scalar::from_u64(k);
            assert!(identity.mul_scalar(&scalar).is_identity(), "k = {k}");
        }

        // A zero scalar sends any point to the identity.
        let zero = Secp256k1Scalar::ZERO;
        assert!(Secp256k1Point::GENERATOR.mul_scalar(&zero).is_identity());
    }
}
