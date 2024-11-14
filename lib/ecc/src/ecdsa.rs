use axvm::intrinsics::IntMod;
use ecdsa::{RecoveryId, Result, Signature, SignatureSize, VerifyingKey};
use elliptic_curve::{
    generic_array::{ArrayLength, GenericArray},
    sec1::{FromEncodedPoint, ModulusSize, ToEncodedPoint},
    AffinePoint, CurveArithmetic, FieldBytes, FieldBytesSize, PrimeCurve, Scalar,
};

use crate::{msm, sw::SwPoint};

pub struct AxvmVerifyingKey<C>(VerifyingKey<C>)
where
    C: PrimeCurve + CurveArithmetic;

impl<C> AxvmVerifyingKey<C>
where
    C: PrimeCurve + CurveArithmetic,
    SignatureSize<C>: ArrayLength<u8>,
    AffinePoint<C>: FromEncodedPoint<C> + ToEncodedPoint<C>,
    FieldBytesSize<C>: ModulusSize,
{
    pub fn recover_from_prehash<Scalar: IntMod>(
        prehash: &[u8],
        sig: &Signature<C>,
        recovery_id: RecoveryId,
    ) {
        let (r, s) = sig.split_scalars();
        let r = Scalar::from_scalar::<C>(r.as_ref());

        // z: IntMod = prehash mod curve order

        // Deal with recovery id

        // Recover the key

        // verify_prehash(vk, prehash, sig)
    }

    pub fn verify_prehashed<Scalar: IntMod, Point: SwPoint>(
        &self,
        prehash: &[u8],
        sig: &Signature<C>,
    ) -> Result<()> {
        let z = Scalar::from_le_bytes(prehash);
        let (r, s) = sig.split_scalars();
        let r = Scalar::from_scalar::<C>(r.as_ref());
        let s = Scalar::from_scalar::<C>(s.as_ref());
        let s_inv = Scalar::ONE.div_unsafe(s); // should IntMod have inv_unsafe?
        let u1 = z * &s_inv;
        let u2 = r * &s_inv;

        let g = Point::generator();
        let q = Point::from_encoded_point::<C>(&self.0.to_encoded_point(false));
        let result = msm(&[u1, u2], &[g, q]);

        Ok(())
    }
}
