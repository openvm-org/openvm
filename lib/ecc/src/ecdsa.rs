use core::ops::{Add, Div};

use axvm_algebra::{IntMod, Reduce};
use ecdsa::{
    hazmat::bits2field, Error, RecoveryId, Result, Signature, SignatureSize, VerifyingKey,
};
use elliptic_curve::{
    bigint::CheckedAdd,
    generic_array::ArrayLength,
    point::DecompressPoint,
    sec1::{FromEncodedPoint, ModulusSize, ToEncodedPoint},
    AffinePoint, CurveArithmetic, FieldBytes, FieldBytesEncoding, FieldBytesSize, PrimeCurve,
    PrimeField,
};

use crate::{msm, sw::SwPoint, CyclicGroup};

// TODO: maybe do IntrinsicCurve: https://github.com/axiom-crypto/afs-prototype/pull/813#discussion_r1847477785
pub struct AxvmVerifyingKey<C>(pub VerifyingKey<C>)
where
    C: PrimeCurve + CurveArithmetic;

impl<C> AxvmVerifyingKey<C>
where
    C: PrimeCurve + CurveArithmetic,
    SignatureSize<C>: ArrayLength<u8>,
    AffinePoint<C>: FromEncodedPoint<C> + ToEncodedPoint<C> + DecompressPoint<C>,
    FieldBytesSize<C>: ModulusSize,
{
    // Ref: https://docs.rs/ecdsa/latest/src/ecdsa/recovery.rs.html#281-316
    #[allow(non_snake_case)]
    pub fn recover_from_prehash<Scalar: IntMod + Reduce, Point: SwPoint + CyclicGroup>(
        prehash: &[u8],
        sig: &Signature<C>,
        recovery_id: RecoveryId,
    ) -> Result<AxvmVerifyingKey<C>>
    where
        for<'a> &'a Point: Add<&'a Point, Output = Point>,
        for<'a> &'a Scalar: Div<&'a Scalar, Output = Scalar>,
    {
        let (r, s) = sig.split_scalars();

        let z = Scalar::from_be_bytes(bits2field::<C>(prehash).unwrap().as_ref());

        let mut r_bytes = r.to_repr();
        if recovery_id.is_x_reduced() {
            // TODO: maybe need to optimize this.
            match Option::<C::Uint>::from(
                C::Uint::decode_field_bytes(&r_bytes).checked_add(&C::ORDER),
            ) {
                Some(restored) => r_bytes = restored.encode_field_bytes(),
                // No reduction should happen here if r was reduced
                None => return Err(Error::new()),
            };
        }
        let R = AffinePoint::<C>::decompress(&r_bytes, u8::from(recovery_id.is_y_odd()).into());

        if R.is_none().into() {
            return Err(Error::new());
        }
        let R = Point::from_encoded_point::<C>(&R.unwrap().to_encoded_point(false));

        let r = Scalar::from_be_bytes(Into::<FieldBytes<C>>::into(r).as_ref());
        let s = Scalar::from_be_bytes(Into::<FieldBytes<C>>::into(s).as_ref());
        let neg_u1 = &z / &r;
        let u2 = &s / &r;
        let NEG_G = Point::NEG_GENERATOR;
        let public_key = msm(&[neg_u1, u2], &[NEG_G, R]);

        let vk = AxvmVerifyingKey(
            VerifyingKey::<C>::from_sec1_bytes(&public_key.to_sec1_bytes(true)).unwrap(),
        );

        vk.verify_prehashed(prehash, sig)?;

        Ok(vk)
    }

    // Ref: https://docs.rs/ecdsa/latest/src/ecdsa/hazmat.rs.html#270
    #[allow(non_snake_case)]
    pub fn verify_prehashed<Scalar: IntMod + Reduce, Point: SwPoint + CyclicGroup>(
        &self,
        prehash: &[u8],
        sig: &Signature<C>,
    ) -> Result<()>
    where
        for<'a> &'a Point: Add<&'a Point, Output = Point>,
        for<'a> &'a Scalar: Div<&'a Scalar, Output = Scalar>,
    {
        let z = Scalar::from_be_bytes(bits2field::<C>(prehash).unwrap().as_ref());
        let (r, s) = sig.split_scalars();
        let r = Scalar::from_be_bytes(Into::<FieldBytes<C>>::into(r).as_ref());
        let s = Scalar::from_be_bytes(Into::<FieldBytes<C>>::into(s).as_ref());
        let u1 = &z / &s;
        let u2 = &r / &s;

        let G = Point::GENERATOR;
        let Q = Point::from_encoded_point::<C>(&self.0.to_encoded_point(false));
        let result = msm(&[u1, u2], &[G, Q]);

        let x_in_scalar = Scalar::reduce_le_bytes(result.x().as_le_bytes());
        if x_in_scalar == r {
            Ok(())
        } else {
            Err(Error::new())
        }
    }
}
