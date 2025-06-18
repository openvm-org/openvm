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
    internal::{Secp256k1Coord, Secp256k1Point, Secp256k1Scalar},
    EncodedPoint, Secp256k1,
};

// --- Implement elliptic_curve traits on Secp256k1Point ---

/// secp256k1 (K-256) field element serialized as bytes.
///
/// Byte array containing a serialized field element value (base field or scalar).
pub type FieldBytes = elliptic_curve::FieldBytes<Secp256k1>;

impl FieldBytesEncoding<Secp256k1> for U256 {
    fn decode_field_bytes(field_bytes: &FieldBytes) -> Self {
        U256::from_be_byte_array(*field_bytes)
    }

    fn encode_field_bytes(&self) -> FieldBytes {
        self.to_be_byte_array()
    }
}

impl AffineCoordinates for Secp256k1Point {
    type FieldRepr = FieldBytes;

    fn x(&self) -> FieldBytes {
        *FieldBytes::from_slice(&<Self as WeierstrassPoint>::x(self).to_be_bytes())
    }

    fn y_is_odd(&self) -> Choice {
        (self.y().as_le_bytes()[0] & 1).into()
    }
}

impl Copy for Secp256k1Point {}

impl ConditionallySelectable for Secp256k1Point {
    fn conditional_select(
        a: &Secp256k1Point,
        b: &Secp256k1Point,
        choice: Choice,
    ) -> Secp256k1Point {
        Secp256k1Point::from_xy_unchecked(
            Secp256k1Coord::conditional_select(
                <Self as WeierstrassPoint>::x(a),
                <Self as WeierstrassPoint>::x(b),
                choice,
            ),
            Secp256k1Coord::conditional_select(a.y(), b.y(), choice),
        )
    }
}

impl ConstantTimeEq for Secp256k1Point {
    fn ct_eq(&self, other: &Secp256k1Point) -> Choice {
        <Self as WeierstrassPoint>::x(self).ct_eq(<Self as WeierstrassPoint>::x(other))
            & self.y().ct_eq(other.y())
    }
}

impl Default for Secp256k1Point {
    fn default() -> Self {
        <Self as WeierstrassPoint>::IDENTITY
    }
}

impl DefaultIsZeroes for Secp256k1Point {}

impl Sum for Secp256k1Point {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(<Self as WeierstrassPoint>::IDENTITY, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Secp256k1Point> for Secp256k1Point {
    fn sum<I: Iterator<Item = &'a Secp256k1Point>>(iter: I) -> Self {
        iter.cloned().sum()
    }
}

impl Mul<Secp256k1Scalar> for Secp256k1Point {
    type Output = Secp256k1Point;

    fn mul(self, other: Secp256k1Scalar) -> Secp256k1Point {
        Secp256k1::msm(&[other], &[self])
    }
}

impl Mul<&Secp256k1Scalar> for &Secp256k1Point {
    type Output = Secp256k1Point;

    fn mul(self, other: &Secp256k1Scalar) -> Secp256k1Point {
        Secp256k1::msm(&[*other], &[*self])
    }
}

impl Mul<&Secp256k1Scalar> for Secp256k1Point {
    type Output = Secp256k1Point;

    fn mul(self, other: &Secp256k1Scalar) -> Secp256k1Point {
        Secp256k1::msm(&[*other], &[self])
    }
}

impl MulAssign<Secp256k1Scalar> for Secp256k1Point {
    fn mul_assign(&mut self, rhs: Secp256k1Scalar) {
        *self = Secp256k1::msm(&[rhs], &[*self]);
    }
}

impl MulAssign<&Secp256k1Scalar> for Secp256k1Point {
    fn mul_assign(&mut self, rhs: &Secp256k1Scalar) {
        *self = Secp256k1::msm(&[*rhs], &[*self]);
    }
}

impl elliptic_curve::Group for Secp256k1Point {
    type Scalar = Secp256k1Scalar;

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

impl elliptic_curve::group::Curve for Secp256k1Point {
    type AffineRepr = Secp256k1Point;

    fn to_affine(&self) -> Secp256k1Point {
        *self
    }
}

impl LinearCombination for Secp256k1Point {
    fn lincomb(x: &Self, k: &Self::Scalar, y: &Self, l: &Self::Scalar) -> Self {
        Secp256k1::msm(&[*k, *l], &[*x, *y])
    }
}

// default implementation
impl MulByGenerator for Secp256k1Point {}

impl DecompressPoint<Secp256k1> for Secp256k1Point {
    /// Note that this is not constant time
    fn decompress(x_bytes: &FieldBytes, y_is_odd: Choice) -> CtOption<Self> {
        use openvm_ecc_guest::weierstrass::FromCompressed;

        let x = Secp256k1Coord::from_be_bytes_unchecked(x_bytes.as_slice());
        let rec_id = y_is_odd.unwrap_u8();
        CtOption::new(x, (x.is_reduced() as u8).into()).and_then(|x| {
            let y = <Secp256k1Point as FromCompressed<Secp256k1Coord>>::decompress(x, &rec_id);
            match y {
                Some(point) => CtOption::new(point, 1.into()),
                None => CtOption::new(Secp256k1Point::default(), 0.into()),
            }
        })
    }
}

// Taken from https://docs.rs/k256/latest/src/k256/arithmetic/affine.rs.html#207
impl DecompactPoint<Secp256k1> for Secp256k1Point {
    fn decompact(x_bytes: &FieldBytes) -> CtOption<Self> {
        Self::decompress(x_bytes, Choice::from(0))
    }
}

impl FromEncodedPoint<Secp256k1> for Secp256k1Point {
    /// Attempts to parse the given [`EncodedPoint`] as an SEC1-encoded [`Secp256k1Point`].
    ///
    /// # Returns
    ///
    /// `None` value if `encoded_point` is not on the secp256k1 curve.
    fn from_encoded_point(encoded_point: &EncodedPoint) -> CtOption<Self> {
        match openvm_ecc_guest::ecdsa::VerifyingKey::<Secp256k1>::from_sec1_bytes(
            encoded_point.as_bytes(),
        ) {
            Ok(verifying_key) => CtOption::new(*verifying_key.as_affine(), 1.into()),
            Err(_) => CtOption::new(Secp256k1Point::default(), 0.into()),
        }
    }
}

impl ToEncodedPoint<Secp256k1> for Secp256k1Point {
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

impl TryFrom<EncodedPoint> for Secp256k1Point {
    type Error = elliptic_curve::Error;

    fn try_from(point: EncodedPoint) -> elliptic_curve::Result<Secp256k1Point> {
        Secp256k1Point::try_from(&point)
    }
}

impl TryFrom<&EncodedPoint> for Secp256k1Point {
    type Error = elliptic_curve::Error;

    fn try_from(point: &EncodedPoint) -> elliptic_curve::Result<Secp256k1Point> {
        Option::from(Secp256k1Point::from_encoded_point(point)).ok_or(elliptic_curve::Error)
    }
}

impl From<Secp256k1Point> for EncodedPoint {
    fn from(affine_point: Secp256k1Point) -> EncodedPoint {
        EncodedPoint::from(&affine_point)
    }
}

impl From<&Secp256k1Point> for EncodedPoint {
    fn from(affine_point: &Secp256k1Point) -> EncodedPoint {
        affine_point.to_encoded_point(true)
    }
}
