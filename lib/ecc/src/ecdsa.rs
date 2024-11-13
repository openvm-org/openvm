use axvm::intrinsics::IntMod;
use ecdsa::{RecoveryId, Result, Signature, SignatureSize, VerifyingKey};
use elliptic_curve::{
    generic_array::{ArrayLength, GenericArray},
    CurveArithmetic, FieldBytes, PrimeCurve, Scalar,
};

pub struct AxvmVerifyingKey<C>(VerifyingKey<C>)
where
    C: PrimeCurve + CurveArithmetic,
    SignatureSize<C>: ArrayLength<u8>;

impl<C> AxvmVerifyingKey<C>
where
    C: PrimeCurve + CurveArithmetic,
    SignatureSize<C>: ArrayLength<u8>,
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

    pub fn verify_prehashed<Scalar: IntMod>(
        &self,
        prehash: &[u8],
        sig: &Signature<C>,
    ) -> Result<()> {
        let z = Scalar::from_le_bytes(prehash);
        let (r, s) = sig.split_scalars();
        let r = Scalar::from_scalar::<C>(r.as_ref());
        let s = Scalar::from_scalar::<C>(s.as_ref());

        Ok(())
    }
}
