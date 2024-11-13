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
    pub fn recover_from_prehash(prehash: &[u8], sig: &Signature<C>, recovery_id: RecoveryId) {
        let (r, s) = sig.split_scalars();
        let r_bytes: FieldBytes<C> = r.into();
        let x: &[u8] = r_bytes.as_slice(); // bytes

        // hmm.. how to do IntMod / Group, that corresponds to C ?? maybe just use generic type.

        // z: IntMod = prehash mod curve order

        // Deal with recovery id

        // Recover the key

        // verify_prehash(vk, prehash, sig)
    }

    pub fn verify_prehashed(&self, prehash: &[u8], sig: &Signature<C>) -> Result<()> {
        Ok(())
    }
}
