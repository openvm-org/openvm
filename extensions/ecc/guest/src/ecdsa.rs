use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul};

use ecdsa::{self, hazmat::bits2field, Error, RecoveryId, Result};
use elliptic_curve::{sec1::Tag, PrimeCurve};
use openvm_algebra_guest::{DivUnsafe, IntMod, Reduce};

use crate::{
    weierstrass::{FromCompressed, IntrinsicCurve, WeierstrassPoint},
    CyclicGroup, Group,
};

pub type Coordinate<C> = <<C as IntrinsicCurve>::Point as WeierstrassPoint>::Coordinate;
pub type Scalar<C> = <C as IntrinsicCurve>::Scalar;

#[repr(C)]
#[derive(Clone)]
pub struct VerifyingKey<C: IntrinsicCurve> {
    pub(crate) inner: PublicKey<C>,
}

#[repr(C)]
#[derive(Clone)]
pub struct PublicKey<C: IntrinsicCurve> {
    /// Affine point
    point: <C as IntrinsicCurve>::Point,
}

impl<C: IntrinsicCurve> PublicKey<C>
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
                let point = FromCompressed::decompress(x, &rec_id);
                Ok(Self { point })
            }

            Tag::Uncompressed => {
                let (x_bytes, y_bytes) = bytes[1..].split_at(Coordinate::<C>::NUM_LIMBS);
                let x = Coordinate::<C>::from_be_bytes(x_bytes);
                let y = Coordinate::<C>::from_be_bytes(y_bytes);
                let point = <C as IntrinsicCurve>::Point::from_xy(x, y).unwrap();
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
}

impl<C> VerifyingKey<C>
where
    C: PrimeCurve + IntrinsicCurve,
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
    ) -> VerifyingKey<C>
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
        // The PartialEq implementation of Scalar: IntMod will constrain `r, s`
        // are in the canonical unique form (i.e., less than the modulus).
        assert_ne!(r, Scalar::<C>::ZERO);
        assert_ne!(s, Scalar::<C>::ZERO);

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
        let R: C::Point = FromCompressed::decompress(x, &rec_id);

        let neg_u1 = z.div_unsafe(&r);
        let u2 = s.div_unsafe(&r);
        let NEG_G = C::Point::NEG_GENERATOR;
        let point = <C as IntrinsicCurve>::msm(&[neg_u1, u2], &[NEG_G, R]);
        let public_key = PublicKey { point };

        VerifyingKey { inner: public_key }
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
        // The PartialEq implementation of Scalar: IntMod will constrain `r, s`
        // are in the canonical unique form (i.e., less than the modulus).
        assert_ne!(r, Scalar::<C>::ZERO);
        assert_ne!(s, Scalar::<C>::ZERO);

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
        let x_mod_n = Scalar::<C>::reduce_le_bytes(x_1.as_le_bytes());
        if x_mod_n == r {
            Ok(())
        } else {
            Err(Error::new())
        }
    }
}
