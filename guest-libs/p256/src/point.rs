use core::{
    iter::Sum,
    ops::{Mul, MulAssign},
};

use elliptic_curve::{
    bigint::{ArrayEncoding, U256},
    ops::{LinearCombination, MulByGenerator},
    point::{AffineCoordinates, DecompactPoint, DecompressPoint},
    rand_core::RngCore,
    sec1::{FromEncodedPoint, ToEncodedPoint},
    subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption},
    zeroize::DefaultIsZeroes,
    FieldBytesEncoding,
};
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{
    weierstrass::{IntrinsicCurve, WeierstrassPoint},
    CyclicGroup,
};

use crate::{
    internal::{P256Coord, P256Point, P256Scalar},
    EncodedPoint, NistP256,
};

// --- Implement elliptic_curve traits on P256Point ---

/// P256 field element serialized as bytes.
///
/// Byte array containing a serialized field element value (base field or scalar).
pub type FieldBytes = elliptic_curve::FieldBytes<NistP256>;

impl FieldBytesEncoding<NistP256> for U256 {
    fn decode_field_bytes(field_bytes: &FieldBytes) -> Self {
        U256::from_be_byte_array(*field_bytes)
    }

    fn encode_field_bytes(&self) -> FieldBytes {
        self.to_be_byte_array()
    }
}

impl AffineCoordinates for P256Point {
    type FieldRepr = FieldBytes;

    fn x(&self) -> FieldBytes {
        *FieldBytes::from_slice(&<Self as WeierstrassPoint>::x(self).to_be_bytes())
    }

    fn y_is_odd(&self) -> Choice {
        (self.y().as_le_bytes()[0] & 1).into()
    }
}

impl Copy for P256Point {}

impl ConditionallySelectable for P256Point {
    fn conditional_select(a: &P256Point, b: &P256Point, choice: Choice) -> P256Point {
        P256Point::from_xy_unchecked(
            P256Coord::conditional_select(
                <Self as WeierstrassPoint>::x(a),
                <Self as WeierstrassPoint>::x(b),
                choice,
            ),
            P256Coord::conditional_select(a.y(), b.y(), choice),
        )
    }
}

impl ConstantTimeEq for P256Point {
    fn ct_eq(&self, other: &P256Point) -> Choice {
        <Self as WeierstrassPoint>::x(self).ct_eq(<Self as WeierstrassPoint>::x(other))
            & self.y().ct_eq(other.y())
    }
}

impl Default for P256Point {
    fn default() -> Self {
        <Self as WeierstrassPoint>::IDENTITY
    }
}

impl DefaultIsZeroes for P256Point {}

impl Sum for P256Point {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(<Self as WeierstrassPoint>::IDENTITY, |a, b| a + b)
    }
}

impl<'a> Sum<&'a P256Point> for P256Point {
    fn sum<I: Iterator<Item = &'a P256Point>>(iter: I) -> Self {
        iter.cloned().sum()
    }
}

impl Mul<P256Scalar> for P256Point {
    type Output = P256Point;

    fn mul(self, other: P256Scalar) -> P256Point {
        NistP256::msm(&[other], &[self])
    }
}

impl Mul<&P256Scalar> for &P256Point {
    type Output = P256Point;

    fn mul(self, other: &P256Scalar) -> P256Point {
        NistP256::msm(&[*other], &[*self])
    }
}

impl Mul<&P256Scalar> for P256Point {
    type Output = P256Point;

    fn mul(self, other: &P256Scalar) -> P256Point {
        NistP256::msm(&[*other], &[self])
    }
}

impl MulAssign<P256Scalar> for P256Point {
    fn mul_assign(&mut self, rhs: P256Scalar) {
        *self = NistP256::msm(&[rhs], &[*self]);
    }
}

impl MulAssign<&P256Scalar> for P256Point {
    fn mul_assign(&mut self, rhs: &P256Scalar) {
        *self = NistP256::msm(&[*rhs], &[*self]);
    }
}

impl elliptic_curve::Group for P256Point {
    type Scalar = P256Scalar;

    fn random(mut _rng: impl RngCore) -> Self {
        // Self::GENERATOR * Self::Scalar::random(&mut rng)
        unimplemented!()
    }

    fn identity() -> Self {
        <Self as WeierstrassPoint>::IDENTITY
    }

    fn generator() -> Self {
        Self::GENERATOR
    }

    fn is_identity(&self) -> Choice {
        (<Self as openvm_ecc_guest::Group>::is_identity(self) as u8).into()
    }

    #[must_use]
    fn double(&self) -> Self {
        self + self
    }
}

impl elliptic_curve::group::Curve for P256Point {
    type AffineRepr = P256Point;

    fn to_affine(&self) -> P256Point {
        *self
    }
}

impl LinearCombination for P256Point {
    fn lincomb(x: &Self, k: &Self::Scalar, y: &Self, l: &Self::Scalar) -> Self {
        NistP256::msm(&[*k, *l], &[*x, *y])
    }
}

// default implementation
impl MulByGenerator for P256Point {}

impl DecompressPoint<NistP256> for P256Point {
    /// Note that this is not constant time
    fn decompress(x_bytes: &FieldBytes, y_is_odd: Choice) -> CtOption<Self> {
        use openvm_ecc_guest::weierstrass::FromCompressed;

        let x = P256Coord::from_be_bytes_unchecked(x_bytes.as_slice());
        let rec_id = y_is_odd.unwrap_u8();
        CtOption::new(x, (x.is_reduced() as u8).into()).and_then(|x| {
            let y = <P256Point as FromCompressed<P256Coord>>::decompress(x, &rec_id);
            match y {
                Some(point) => CtOption::new(point, 1.into()),
                None => CtOption::new(P256Point::default(), 0.into()),
            }
        })
    }
}

impl DecompactPoint<NistP256> for P256Point {
    fn decompact(x_bytes: &FieldBytes) -> CtOption<Self> {
        Self::decompress(x_bytes, Choice::from(0))
    }
}

impl FromEncodedPoint<NistP256> for P256Point {
    /// Attempts to parse the given [`EncodedPoint`] as an SEC1-encoded [`P256Point`].
    ///
    /// # Returns
    ///
    /// `None` value if `encoded_point` is not on the secp256k1 curve.
    fn from_encoded_point(encoded_point: &EncodedPoint) -> CtOption<Self> {
        match openvm_ecc_guest::ecdsa::VerifyingKey::<NistP256>::from_sec1_bytes(
            encoded_point.as_bytes(),
        ) {
            Ok(verifying_key) => CtOption::new(*verifying_key.as_affine(), 1.into()),
            Err(_) => CtOption::new(P256Point::default(), 0.into()),
        }
    }
}

impl ToEncodedPoint<NistP256> for P256Point {
    fn to_encoded_point(&self, compress: bool) -> EncodedPoint {
        EncodedPoint::conditional_select(
            &EncodedPoint::from_affine_coordinates(
                &<Self as WeierstrassPoint>::x(self).to_be_bytes().into(),
                &<Self as WeierstrassPoint>::y(self).to_be_bytes().into(),
                compress,
            ),
            &EncodedPoint::identity(),
            elliptic_curve::Group::is_identity(self),
        )
    }
}

impl TryFrom<EncodedPoint> for P256Point {
    type Error = elliptic_curve::Error;

    fn try_from(point: EncodedPoint) -> elliptic_curve::Result<P256Point> {
        P256Point::try_from(&point)
    }
}

impl TryFrom<&EncodedPoint> for P256Point {
    type Error = elliptic_curve::Error;

    fn try_from(point: &EncodedPoint) -> elliptic_curve::Result<P256Point> {
        Option::from(P256Point::from_encoded_point(point)).ok_or(elliptic_curve::Error)
    }
}

impl From<P256Point> for EncodedPoint {
    fn from(affine_point: P256Point) -> EncodedPoint {
        EncodedPoint::from(&affine_point)
    }
}

impl From<&P256Point> for EncodedPoint {
    fn from(affine_point: &P256Point) -> EncodedPoint {
        affine_point.to_encoded_point(true)
    }
}
