use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul};

use ecdsa_core::{
    self,
    hazmat::{bits2field, DigestPrimitive},
    signature::{
        digest::{Digest, FixedOutput},
        hazmat::PrehashVerifier,
        DigestVerifier, Verifier,
    },
    EncodedPoint, Error, RecoveryId, Result, Signature, SignatureSize,
};
use elliptic_curve::{
    generic_array::ArrayLength,
    sec1::{FromEncodedPoint, ModulusSize, Tag, ToEncodedPoint},
    subtle::{Choice, ConditionallySelectable, CtOption},
    CurveArithmetic, FieldBytesSize, PrimeCurve,
};
use openvm_algebra_guest::{DivUnsafe, IntMod, Reduce};

use crate::{
    weierstrass::{FromCompressed, IntrinsicCurve, WeierstrassPoint},
    CyclicGroup, Group,
};

type Coordinate<C> = <<C as IntrinsicCurve>::Point as WeierstrassPoint>::Coordinate;
type Scalar<C> = <C as IntrinsicCurve>::Scalar;
type AffinePoint<C> = <C as IntrinsicCurve>::Point;

//
// Signing implementations are placeholders to support patching compilation
//

/// This is placeholder struct for compatibility purposes with the `ecdsa` crate.
/// Signing from private keys is not supported yet.
#[derive(Clone)]
pub struct SigningKey<C: IntrinsicCurve> {
    /// ECDSA signing keys are non-zero elements of a given curve's scalar field.
    #[allow(dead_code)]
    secret_scalar: NonZeroScalar<C>,

    /// Verifying key which corresponds to this signing key.
    verifying_key: VerifyingKey<C>,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct NonZeroScalar<C: IntrinsicCurve> {
    scalar: Scalar<C>,
}

impl<C: IntrinsicCurve> SigningKey<C> {
    pub fn from_slice(_bytes: &[u8]) -> Result<Self> {
        todo!("signing is not yet implemented")
    }

    pub fn verifying_key(&self) -> &VerifyingKey<C> {
        &self.verifying_key
    }
}

impl<C> SigningKey<C>
where
    C: IntrinsicCurve + PrimeCurve,
{
    pub fn sign_prehash_recoverable(&self, _prehash: &[u8]) -> Result<(Signature<C>, RecoveryId)> {
        todo!("signing is not yet implemented")
    }
}

// This struct is public because it is used by the VerifyPrimitive impl in the k256 and p256 guest
// libraries.
#[repr(C)]
#[derive(Clone)]
pub struct VerifyingKey<C: IntrinsicCurve> {
    pub(crate) inner: PublicKey<C>,
}

// This struct is public because it is used by the VerifyPrimitive impl in the k256 and p256 guest
#[repr(C)]
#[derive(Clone)]
pub struct PublicKey<C: IntrinsicCurve> {
    /// Affine point
    point: AffinePoint<C>,
}

impl<C: IntrinsicCurve> PublicKey<C>
where
    C::Point: WeierstrassPoint + Group + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
{
    pub fn new(point: <C as IntrinsicCurve>::Point) -> Self {
        Self { point }
    }

    pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self>
    where
        for<'a> &'a Coordinate<C>: Mul<&'a Coordinate<C>, Output = Coordinate<C>>,
    {
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

    pub fn as_affine(&self) -> &AffinePoint<C> {
        &self.point
    }

    pub fn into_affine(self) -> AffinePoint<C> {
        self.point
    }
}

impl<C: IntrinsicCurve> VerifyingKey<C>
where
    C::Point: WeierstrassPoint + Group + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    for<'a> &'a Coordinate<C>: Mul<&'a Coordinate<C>, Output = Coordinate<C>>,
{
    pub fn new(public_key: PublicKey<C>) -> Self {
        Self { inner: public_key }
    }

    pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self> {
        let public_key = PublicKey::<C>::from_sec1_bytes(bytes)?;
        Ok(Self::new(public_key))
    }

    pub fn from_affine(point: <C as IntrinsicCurve>::Point) -> Result<Self> {
        let public_key = PublicKey::<C>::new(point);
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

// Functions for compatibility with `ecdsa` crate
impl<C> VerifyingKey<C>
where
    C: IntrinsicCurve + PrimeCurve,
    C::Point: WeierstrassPoint + CyclicGroup + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    C::Scalar: IntMod + Reduce,
    for<'a> &'a C::Point: Add<&'a C::Point, Output = C::Point>,
    for<'a> &'a Coordinate<C>: Mul<&'a Coordinate<C>, Output = Coordinate<C>>,
    FieldBytesSize<C>: ModulusSize,
    SignatureSize<C>: ArrayLength<u8>,
{
    /// Recover a [`VerifyingKey`] from the given message, signature, and
    /// [`RecoveryId`].
    ///
    /// The message is first hashed using this curve's [`DigestPrimitive`].
    pub fn recover_from_msg(
        msg: &[u8],
        signature: &Signature<C>,
        recovery_id: RecoveryId,
    ) -> Result<Self>
    where
        C: DigestPrimitive,
    {
        Self::recover_from_digest(C::Digest::new_with_prefix(msg), signature, recovery_id)
    }

    /// Recover a [`VerifyingKey`] from the given message [`Digest`],
    /// signature, and [`RecoveryId`].
    pub fn recover_from_digest<D>(
        msg_digest: D,
        signature: &Signature<C>,
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
    pub fn recover_from_prehash(
        prehash: &[u8],
        signature: &Signature<C>,
        recovery_id: RecoveryId,
    ) -> Result<Self> {
        let sig = signature.to_bytes();
        Self::recover_from_prehash_noverify(prehash, &sig, recovery_id)
    }
}

//
// `*Verifier` trait impls
//

impl<C, D> DigestVerifier<D, Signature<C>> for VerifyingKey<C>
where
    C: PrimeCurve + IntrinsicCurve,
    D: Digest + FixedOutput<OutputSize = FieldBytesSize<C>>,
    SignatureSize<C>: ArrayLength<u8>,
    C::Point: WeierstrassPoint + CyclicGroup + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    <C as IntrinsicCurve>::Scalar: IntMod + Reduce,
    for<'a> &'a C::Point: Add<&'a C::Point, Output = C::Point>,
    for<'a> &'a Scalar<C>: DivUnsafe<&'a Scalar<C>, Output = Scalar<C>>,
{
    fn verify_digest(&self, msg_digest: D, signature: &Signature<C>) -> Result<()> {
        verify_prehashed::<C>(
            self.inner.as_affine().clone(),
            &msg_digest.finalize_fixed(),
            &signature.to_bytes(),
        )
    }
}

impl<C> PrehashVerifier<Signature<C>> for VerifyingKey<C>
where
    C: PrimeCurve + IntrinsicCurve,
    SignatureSize<C>: ArrayLength<u8>,
    C::Point: WeierstrassPoint + CyclicGroup + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    C::Scalar: IntMod + Reduce,
    for<'a> &'a C::Point: Add<&'a C::Point, Output = C::Point>,
    for<'a> &'a Scalar<C>: DivUnsafe<&'a Scalar<C>, Output = Scalar<C>>,
{
    fn verify_prehash(&self, prehash: &[u8], signature: &Signature<C>) -> Result<()> {
        verify_prehashed::<C>(
            self.inner.as_affine().clone(),
            prehash,
            &signature.to_bytes(),
        )
    }
}

impl<C> Verifier<Signature<C>> for VerifyingKey<C>
where
    C: PrimeCurve + CurveArithmetic + DigestPrimitive + IntrinsicCurve,
    SignatureSize<C>: ArrayLength<u8>,
    C::Point: WeierstrassPoint + CyclicGroup + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    <C as IntrinsicCurve>::Scalar: IntMod + Reduce,
    for<'a> &'a C::Point: Add<&'a C::Point, Output = C::Point>,
    for<'a> &'a Scalar<C>: DivUnsafe<&'a Scalar<C>, Output = Scalar<C>>,
{
    fn verify(&self, msg: &[u8], signature: &Signature<C>) -> Result<()> {
        self.verify_digest(C::Digest::new_with_prefix(msg), signature)
    }
}

//
// copied from `ecdsa`
//
impl<C> VerifyingKey<C>
where
    C: CurveArithmetic + IntrinsicCurve,
    AffinePoint<C>: FromEncodedPoint<C> + ToEncodedPoint<C> + Default + ConditionallySelectable,
    FieldBytesSize<C>: ModulusSize,
{
    /// Initialize [`VerifyingKey`] from an [`EncodedPoint`].
    pub fn from_encoded_point(public_key: &EncodedPoint<C>) -> Result<Self> {
        Option::from(PublicKey::<C>::from_encoded_point(public_key))
            .map(|public_key| Self { inner: public_key })
            .ok_or_else(Error::new)
    }

    /// Serialize this [`VerifyingKey`] as a SEC1 [`EncodedPoint`], optionally
    /// applying point compression.
    pub fn to_encoded_point(&self, compress: bool) -> EncodedPoint<C> {
        self.inner.to_encoded_point(compress)
    }
}

//
// sec1 traits copied from elliptic_curve
//
impl<C> FromEncodedPoint<C> for PublicKey<C>
where
    C: CurveArithmetic + IntrinsicCurve,
    AffinePoint<C>: FromEncodedPoint<C> + ToEncodedPoint<C> + Default + ConditionallySelectable,
    FieldBytesSize<C>: ModulusSize,
{
    /// Initialize [`PublicKey`] from an [`EncodedPoint`]
    fn from_encoded_point(encoded_point: &EncodedPoint<C>) -> CtOption<Self> {
        AffinePoint::<C>::from_encoded_point(encoded_point).and_then(|point| {
            // Defeating the point of `subtle`, but the use case is specifically a public key
            let is_identity = Choice::from(u8::from(encoded_point.is_identity()));
            CtOption::new(PublicKey { point }, !is_identity)
        })
    }
}

impl<C> ToEncodedPoint<C> for PublicKey<C>
where
    C: CurveArithmetic + IntrinsicCurve,
    AffinePoint<C>: FromEncodedPoint<C> + ToEncodedPoint<C>,
    FieldBytesSize<C>: ModulusSize,
{
    /// Serialize this [`PublicKey`] as a SEC1 [`EncodedPoint`], optionally applying
    /// point compression
    fn to_encoded_point(&self, compress: bool) -> EncodedPoint<C> {
        self.point.to_encoded_point(compress)
    }
}

// Custom openvm implementations
impl<C> VerifyingKey<C>
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
        let public_key = PublicKey { point };

        Ok(VerifyingKey { inner: public_key })
    }
}

// Ref: https://docs.rs/ecdsa/latest/src/ecdsa/hazmat.rs.html#270
#[allow(non_snake_case)]
pub fn verify_prehashed<C>(pubkey: AffinePoint<C>, prehash: &[u8], sig: &[u8]) -> Result<()>
where
    C: IntrinsicCurve + PrimeCurve,
    C::Point: WeierstrassPoint + CyclicGroup + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    C::Scalar: IntMod + Reduce,
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
    let z =
        <C as IntrinsicCurve>::Scalar::from_be_bytes(bits2field::<C>(prehash).unwrap().as_ref());

    let u1 = z.div_unsafe(&s);
    let u2 = (&r).div_unsafe(&s);

    let G = C::Point::GENERATOR;
    // public key
    let Q = pubkey;
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

impl<C: IntrinsicCurve> AsRef<AffinePoint<C>> for VerifyingKey<C> {
    fn as_ref(&self) -> &AffinePoint<C> {
        &self.inner.point
    }
}

impl<C: IntrinsicCurve> AsRef<AffinePoint<C>> for PublicKey<C> {
    fn as_ref(&self) -> &AffinePoint<C> {
        &self.point
    }
}
