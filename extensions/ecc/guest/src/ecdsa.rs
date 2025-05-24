use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul};

use ecdsa::{
    self,
    hazmat::{bits2field, DigestPrimitive, VerifyPrimitive},
    EncodedPoint, Error, RecoveryId, Result, Signature, SignatureSize,
};
use elliptic_curve::{
    generic_array::ArrayLength,
    point::{DecompressPoint, PointCompression},
    sec1::{FromEncodedPoint, ModulusSize, Tag, ToEncodedPoint},
    AffinePoint, CurveArithmetic, FieldBytesSize, PrimeCurve,
};
use openvm_algebra_guest::{DivUnsafe, IntMod, Reduce};
use signature::{digest::Digest, hazmat::PrehashVerifier};

use crate::{
    weierstrass::{FromCompressed, IntrinsicCurve, WeierstrassPoint},
    CyclicGroup, Group,
};

type Coordinate<C> = <<C as IntrinsicCurve>::Point as WeierstrassPoint>::Coordinate;
type Scalar<C> = <C as IntrinsicCurve>::Scalar;

#[repr(C)]
#[derive(Clone)]
pub struct VerifyingKey<C1, C2: PrimeCurve + CurveArithmetic> {
    pub(crate) ecdsa_verifying_key: ecdsa::VerifyingKey<C2>,
    // C1 is the internal struct associated to the curve specified by C2
    phantom: core::marker::PhantomData<C1>,
}
impl<C1, C2: PrimeCurve + CurveArithmetic> VerifyingKey<C1, C2>
where
    C1: IntrinsicCurve + PrimeCurve,
    C1::Point:
        WeierstrassPoint + CyclicGroup + FromCompressed<Coordinate<C1>> + Into<C2::AffinePoint>,
    Coordinate<C1>: IntMod,
    C1::Scalar: IntMod + Reduce,
    for<'a> &'a C1::Point: Add<&'a C1::Point, Output = C1::Point>,
    for<'a> &'a Coordinate<C1>: Mul<&'a Coordinate<C1>, Output = Coordinate<C1>>,
    C2: PrimeCurve + CurveArithmetic,
    AffinePoint<C2>:
        DecompressPoint<C2> + FromEncodedPoint<C2> + ToEncodedPoint<C2> + VerifyPrimitive<C2>,
    FieldBytesSize<C2>: ModulusSize,
    SignatureSize<C2>: ArrayLength<u8>,
{
    /// Recover a [`VerifyingKey`] from the given message, signature, and
    /// [`RecoveryId`].
    ///
    /// The message is first hashed using this curve's [`DigestPrimitive`].
    pub fn recover_from_msg(
        msg: &[u8],
        signature: &Signature<C2>,
        recovery_id: RecoveryId,
    ) -> Result<Self>
    where
        C2: DigestPrimitive,
    {
        Self::recover_from_digest(C2::Digest::new_with_prefix(msg), signature, recovery_id)
    }

    /// Recover a [`VerifyingKey`] from the given message [`Digest`],
    /// signature, and [`RecoveryId`].
    pub fn recover_from_digest<D>(
        msg_digest: D,
        signature: &Signature<C2>,
        recovery_id: RecoveryId,
    ) -> Result<Self>
    where
        D: Digest,
    {
        Self::recover_from_prehash(&msg_digest.finalize(), signature, recovery_id)
    }

    /// Recover a [`VerifyingKey`] from the given `prehash` of a message, the
    /// signature over that prehashed message, and a [`RecoveryId`].
    /// Note that this function does not verify the signature with the recovered key.
    #[allow(non_snake_case)]
    pub fn recover_from_prehash(
        prehash: &[u8],
        signature: &Signature<C2>,
        recovery_id: RecoveryId,
    ) -> Result<Self> {
        let ret = OpenVMVerifyingKey::<C1>::recover_from_prehash_noverify(
            prehash,
            &signature.to_bytes(),
            recovery_id,
        )?;
        Ok(Self {
            ecdsa_verifying_key: ecdsa::VerifyingKey::<C2>::from_affine(ret.into_affine().into())?,
            phantom: core::marker::PhantomData,
        })
    }
}

// Pass through the functions on the inner ecdsa::VerifyingKey
// See: https://docs.rs/ecdsa/0.16.9/src/ecdsa/verifying.rs.html#85
impl<C1, C2: PrimeCurve + CurveArithmetic> VerifyingKey<C1, C2>
where
    C1: IntrinsicCurve + PrimeCurve,
    C1::Point:
        WeierstrassPoint + CyclicGroup + FromCompressed<Coordinate<C1>> + Into<C2::AffinePoint>,
    Coordinate<C1>: IntMod,
    C1::Scalar: IntMod + Reduce,
    for<'a> &'a C1::Point: Add<&'a C1::Point, Output = C1::Point>,
    for<'a> &'a Coordinate<C1>: Mul<&'a Coordinate<C1>, Output = Coordinate<C1>>,
    C2: PrimeCurve + CurveArithmetic,
    AffinePoint<C2>: FromEncodedPoint<C2> + ToEncodedPoint<C2>,
    FieldBytesSize<C2>: ModulusSize,
{
    /// Initialize [`VerifyingKey`] from a SEC1-encoded public key.
    pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            ecdsa_verifying_key: ecdsa::VerifyingKey::<C2>::from_sec1_bytes(bytes)?,
            phantom: core::marker::PhantomData,
        })
    }

    /// Initialize [`VerifyingKey`] from an affine point.
    ///
    /// Returns an [`Error`] if the given affine point is the additive identity
    /// (a.k.a. point at infinity).
    pub fn from_affine(affine: AffinePoint<C2>) -> Result<Self> {
        Ok(Self {
            ecdsa_verifying_key: ecdsa::VerifyingKey::<C2>::from_affine(affine.into())?,
            phantom: core::marker::PhantomData,
        })
    }

    /// Initialize [`VerifyingKey`] from an [`EncodedPoint`].
    pub fn from_encoded_point(public_key: &EncodedPoint<C2>) -> Result<Self> {
        Ok(Self {
            ecdsa_verifying_key: ecdsa::VerifyingKey::<C2>::from_encoded_point(public_key)?,
            phantom: core::marker::PhantomData,
        })
    }

    /// Serialize this [`VerifyingKey`] as a SEC1 [`EncodedPoint`], optionally
    /// applying point compression.
    pub fn to_encoded_point(&self, compress: bool) -> EncodedPoint<C2> {
        self.ecdsa_verifying_key.to_encoded_point(compress)
    }

    /// Convert this [`VerifyingKey`] into the
    /// `Elliptic-Curve-Point-to-Octet-String` encoding described in
    /// SEC 1: Elliptic Curve Cryptography (Version 2.0) section 2.3.3
    /// (page 10).
    ///
    /// <http://www.secg.org/sec1-v2.pdf>
    #[cfg(feature = "alloc")]
    pub fn to_sec1_bytes(&self) -> Box<[u8]>
    where
        C2: PointCompression,
    {
        self.ecdsa_verifying_key.to_sec1_bytes()
    }

    /// Borrow the inner [`AffinePoint`] for this public key.
    pub fn as_affine(&self) -> &AffinePoint<C2> {
        self.ecdsa_verifying_key.as_affine()
    }
}

// From https://docs.rs/ecdsa/0.16.9/src/ecdsa/verifying.rs.html#157
impl<C1, C2: PrimeCurve + CurveArithmetic> PrehashVerifier<Signature<C2>> for VerifyingKey<C1, C2>
where
    C1: IntrinsicCurve + PrimeCurve,
    C2: PrimeCurve + CurveArithmetic,
    AffinePoint<C2>: VerifyPrimitive<C2>,
    SignatureSize<C2>: ArrayLength<u8>,
{
    fn verify_prehash(&self, prehash: &[u8], signature: &Signature<C2>) -> Result<()> {
        self.ecdsa_verifying_key.verify_prehash(prehash, signature)
    }
}

// From https://docs.rs/ecdsa/0.16.9/src/ecdsa/verifying.rs.html#294
impl<C1, C2: PrimeCurve + CurveArithmetic> From<&VerifyingKey<C1, C2>> for EncodedPoint<C2>
where
    C1: IntrinsicCurve + PrimeCurve,
    C2: PrimeCurve + CurveArithmetic + PointCompression,
    AffinePoint<C2>: FromEncodedPoint<C2> + ToEncodedPoint<C2>,
    FieldBytesSize<C2>: ModulusSize,
{
    fn from(verifying_key: &VerifyingKey<C1, C2>) -> EncodedPoint<C2> {
        verifying_key.ecdsa_verifying_key.into()
    }
}

// This struct is public because it is used by the VerifyPrimitive impl in the k256 and p256 guest
// libraries.
#[repr(C)]
#[derive(Clone)]
pub struct OpenVMVerifyingKey<C: IntrinsicCurve> {
    pub(crate) inner: OpenVMPublicKey<C>,
}

// This struct is public because it is used by the VerifyPrimitive impl in the k256 and p256 guest
#[repr(C)]
#[derive(Clone)]
pub struct OpenVMPublicKey<C: IntrinsicCurve> {
    /// Affine point
    point: <C as IntrinsicCurve>::Point,
}

impl<C: IntrinsicCurve> OpenVMPublicKey<C>
where
    C::Point: WeierstrassPoint + Group + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    for<'a> &'a Coordinate<C>: Mul<&'a Coordinate<C>, Output = Coordinate<C>>,
{
    pub fn new(point: <C as IntrinsicCurve>::Point) -> Self {
        Self { point }
    }

    pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(Error::new());
        }

        // Validate tag
        let tag = Tag::from_u8(bytes[0]).unwrap();

        // Validate length
        let expected_len = tag.message_len(Coordinate::<C>::NUM_LIMBS);
        if bytes.len() != expected_len {
            return Err(Error::new());
        }

        match tag {
            Tag::Identity => {
                let point = <<C as IntrinsicCurve>::Point as WeierstrassPoint>::IDENTITY;
                Ok(Self { point })
            }

            Tag::CompressedEvenY | Tag::CompressedOddY => {
                let x = Coordinate::<C>::from_be_bytes(&bytes[1..]);
                let rec_id = bytes[0] & 1;
                let point = FromCompressed::decompress(x, &rec_id).ok_or_else(Error::new)?;
                Ok(Self { point })
            }

            Tag::Uncompressed => {
                let (x_bytes, y_bytes) = bytes[1..].split_at(Coordinate::<C>::NUM_LIMBS);
                let x = Coordinate::<C>::from_be_bytes(x_bytes);
                let y = Coordinate::<C>::from_be_bytes(y_bytes);
                let point = <C as IntrinsicCurve>::Point::from_xy(x, y).ok_or_else(Error::new)?;
                Ok(Self { point })
            }

            _ => Err(Error::new()),
        }
    }

    pub fn to_sec1_bytes(&self, compress: bool) -> Vec<u8> {
        if self.point.is_identity() {
            return vec![0x00];
        }

        let (x, y) = self.point.clone().into_coords();

        if compress {
            let mut bytes = Vec::<u8>::with_capacity(1 + Coordinate::<C>::NUM_LIMBS);
            let tag = if y.as_le_bytes()[0] & 1 == 1 {
                Tag::CompressedOddY
            } else {
                Tag::CompressedEvenY
            };
            bytes.push(tag.into());
            bytes.extend_from_slice(x.to_be_bytes().as_ref());
            bytes
        } else {
            let mut bytes = Vec::<u8>::with_capacity(1 + Coordinate::<C>::NUM_LIMBS * 2);
            bytes.push(Tag::Uncompressed.into());
            bytes.extend_from_slice(x.to_be_bytes().as_ref());
            bytes.extend_from_slice(y.to_be_bytes().as_ref());
            bytes
        }
    }

    pub fn as_affine(&self) -> &<C as IntrinsicCurve>::Point {
        &self.point
    }

    pub fn into_affine(self) -> <C as IntrinsicCurve>::Point {
        self.point
    }
}

impl<C: IntrinsicCurve> OpenVMVerifyingKey<C>
where
    C::Point: WeierstrassPoint + Group + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    for<'a> &'a Coordinate<C>: Mul<&'a Coordinate<C>, Output = Coordinate<C>>,
{
    pub fn new(public_key: OpenVMPublicKey<C>) -> Self {
        Self { inner: public_key }
    }

    pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self> {
        let public_key = OpenVMPublicKey::<C>::from_sec1_bytes(bytes)?;
        Ok(Self::new(public_key))
    }

    pub fn from_affine(point: <C as IntrinsicCurve>::Point) -> Result<Self> {
        let public_key = OpenVMPublicKey::<C>::new(point);
        Ok(Self::new(public_key))
    }

    pub fn to_sec1_bytes(&self, compress: bool) -> Vec<u8> {
        self.inner.to_sec1_bytes(compress)
    }

    pub fn as_affine(&self) -> &<C as IntrinsicCurve>::Point {
        self.inner.as_affine()
    }

    pub fn into_affine(self) -> <C as IntrinsicCurve>::Point {
        self.inner.into_affine()
    }
}

impl<C> OpenVMVerifyingKey<C>
where
    C: IntrinsicCurve + PrimeCurve,
    C::Point: WeierstrassPoint + CyclicGroup + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    C::Scalar: IntMod + Reduce,
{
    /// Ref: <https://github.com/RustCrypto/signatures/blob/85c984bcc9927c2ce70c7e15cbfe9c6936dd3521/ecdsa/src/recovery.rs#L297>
    ///
    /// Recovery does not require additional signature verification: <https://github.com/RustCrypto/signatures/pull/831>
    ///
    /// ## Panics
    /// If the signature is invalid or public key cannot be recovered from the given input.
    #[allow(non_snake_case)]
    pub fn recover_from_prehash_noverify(
        prehash: &[u8],
        sig: &[u8],
        recovery_id: RecoveryId,
    ) -> Result<Self>
    where
        for<'a> &'a C::Point: Add<&'a C::Point, Output = C::Point>,
        for<'a> &'a Coordinate<C>: Mul<&'a Coordinate<C>, Output = Coordinate<C>>,
    {
        // This should get compiled out:
        assert!(Scalar::<C>::NUM_LIMBS <= Coordinate::<C>::NUM_LIMBS);
        // IntMod limbs are currently always bytes
        assert_eq!(sig.len(), <C as IntrinsicCurve>::Scalar::NUM_LIMBS * 2);
        // Signature is default encoded in big endian bytes
        let (r_be, s_be) = sig.split_at(<C as IntrinsicCurve>::Scalar::NUM_LIMBS);
        // Note: Scalar internally stores using little endian
        let r = Scalar::<C>::from_be_bytes(r_be);
        let s = Scalar::<C>::from_be_bytes(s_be);
        if !r.is_reduced() || !s.is_reduced() {
            return Err(Error::new());
        }
        if r == Scalar::<C>::ZERO || s == Scalar::<C>::ZERO {
            return Err(Error::new());
        }

        // Perf: don't use bits2field from ::ecdsa
        let z = Scalar::<C>::from_be_bytes(bits2field::<C>(prehash).unwrap().as_ref());

        // `r` is in the Scalar field, we now possibly add C::ORDER to it to get `x`
        // in the Coordinate field.
        let mut x = Coordinate::<C>::from_le_bytes(r.as_le_bytes());
        if recovery_id.is_x_reduced() {
            // Copy from slice in case Coordinate has more bytes than Scalar
            let order = Coordinate::<C>::from_le_bytes(Scalar::<C>::MODULUS.as_ref());
            x.add_assign(order);
        }
        let rec_id = recovery_id.to_byte();
        // The point R decompressed from x-coordinate `r`
        let R: C::Point = FromCompressed::decompress(x, &rec_id).ok_or_else(Error::new)?;

        let neg_u1 = z.div_unsafe(&r);
        let u2 = s.div_unsafe(&r);
        let NEG_G = C::Point::NEG_GENERATOR;
        let point = <C as IntrinsicCurve>::msm(&[neg_u1, u2], &[NEG_G, R]);
        let public_key = OpenVMPublicKey { point };

        Ok(OpenVMVerifyingKey { inner: public_key })
    }

    // Ref: https://docs.rs/ecdsa/latest/src/ecdsa/hazmat.rs.html#270
    #[allow(non_snake_case)]
    pub fn verify_prehashed(self, prehash: &[u8], sig: &[u8]) -> Result<()>
    where
        for<'a> &'a C::Point: Add<&'a C::Point, Output = C::Point>,
        for<'a> &'a Scalar<C>: DivUnsafe<&'a Scalar<C>, Output = Scalar<C>>,
    {
        // This should get compiled out:
        assert!(Scalar::<C>::NUM_LIMBS <= Coordinate::<C>::NUM_LIMBS);
        // IntMod limbs are currently always bytes
        assert_eq!(sig.len(), Scalar::<C>::NUM_LIMBS * 2);
        // Signature is default encoded in big endian bytes
        let (r_be, s_be) = sig.split_at(<C as IntrinsicCurve>::Scalar::NUM_LIMBS);
        // Note: Scalar internally stores using little endian
        let r = Scalar::<C>::from_be_bytes(r_be);
        let s = Scalar::<C>::from_be_bytes(s_be);
        if !r.is_reduced() || !s.is_reduced() {
            return Err(Error::new());
        }
        if r == Scalar::<C>::ZERO || s == Scalar::<C>::ZERO {
            return Err(Error::new());
        }

        // Perf: don't use bits2field from ::ecdsa
        let z = <C as IntrinsicCurve>::Scalar::from_be_bytes(
            bits2field::<C>(prehash).unwrap().as_ref(),
        );

        let u1 = z.div_unsafe(&s);
        let u2 = (&r).div_unsafe(&s);

        let G = C::Point::GENERATOR;
        // public key
        let Q = self.inner.point;
        let R = <C as IntrinsicCurve>::msm(&[u1, u2], &[G, Q]);
        if R.is_identity() {
            return Err(Error::new());
        }
        let (x_1, _) = R.into_coords();
        // Scalar and Coordinate may be different byte lengths, so we use an inefficient reduction
        let x_mod_n = Scalar::<C>::reduce_le_bytes(x_1.as_le_bytes());
        if x_mod_n == r {
            Ok(())
        } else {
            Err(Error::new())
        }
    }
}
