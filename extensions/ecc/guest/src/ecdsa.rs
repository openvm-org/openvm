use core::ops::{Add, AddAssign, Mul};

use ecdsa::{self, hazmat::bits2field, Error, RecoveryId, Result};
use elliptic_curve::PrimeCurve;
use openvm_algebra_guest::{DivUnsafe, IntMod, Reduce};

use crate::{
    weierstrass::{IntrinsicCurve, WeierstrassPoint},
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

impl<C: IntrinsicCurve> PublicKey<C> {
    pub fn into_inner(self) -> <C as IntrinsicCurve>::Point {
        self.point
    }
}

impl<C: IntrinsicCurve> VerifyingKey<C> {
    pub fn as_affine(&self) -> &<C as IntrinsicCurve>::Point {
        &self.inner.point
    }
}

impl<C> VerifyingKey<C>
where
    C: PrimeCurve + IntrinsicCurve,
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

        // TODO: don't use bits2field from ::ecdsa
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
        let R: C::Point = WeierstrassPoint::decompress(x, &rec_id);

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

        // TODO: don't use bits2field from ::ecdsa
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
