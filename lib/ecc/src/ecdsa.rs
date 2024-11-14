use alloc::{
    format,
    string::{String, ToString},
};
use core::ops::{Add, Mul};

use axvm::{intrinsics::IntMod, io::print};
use ecdsa::{
    hazmat::bits2field, Error, RecoveryId, Result, Signature, SignatureSize, VerifyingKey,
};
use elliptic_curve::{
    bigint::CheckedAdd,
    generic_array::ArrayLength,
    point::DecompressPoint,
    sec1::{FromEncodedPoint, ModulusSize, ToEncodedPoint},
    AffinePoint, CurveArithmetic, FieldBytes, FieldBytesEncoding, FieldBytesSize, PrimeCurve,
    PrimeField, Scalar,
};

use crate::{msm, sw::SwPoint};

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
    pub fn recover_from_prehash<Scalar: IntMod, Point: SwPoint>(
        prehash: &[u8],
        sig: &Signature<C>,
        recovery_id: RecoveryId,
    ) -> Result<AxvmVerifyingKey<C>>
    where
        for<'a> &'a Point: Add<&'a Point, Output = Point>,
        for<'a> &'a Scalar: Mul<&'a Scalar, Output = Scalar>,
    {
        let (r, s) = sig.split_scalars();

        let z = Scalar::from_be_bytes(bits2field::<C>(prehash).unwrap().as_ref());

        let mut r_bytes = r.to_repr();
        if recovery_id.is_x_reduced() {
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
        let r_inv = Scalar::ONE.div_unsafe(r);
        let u1 = -(&z * &r_inv);
        let u2 = &s * &r_inv;
        let G = Point::generator();
        let public_key = msm(&[u1, u2], &[G, R]);

        let vk = AxvmVerifyingKey(
            VerifyingKey::<C>::from_sec1_bytes(&public_key.to_sec1_bytes(true)).unwrap(),
        );

        vk.verify_prehashed(prehash, sig)?;

        Ok(vk)
    }

    // Ref: https://docs.rs/ecdsa/latest/src/ecdsa/hazmat.rs.html#270
    #[allow(non_snake_case)]
    pub fn verify_prehashed<Scalar: IntMod, Point: SwPoint>(
        &self,
        prehash: &[u8],
        sig: &Signature<C>,
    ) -> Result<()>
    where
        for<'a> &'a Point: Add<&'a Point, Output = Point>,
        for<'a> &'a Scalar: Mul<&'a Scalar, Output = Scalar>,
    {
        let z = Scalar::from_be_bytes(bits2field::<C>(prehash).unwrap().as_ref());
        let (r, s) = sig.split_scalars();
        let r = Scalar::from_be_bytes(Into::<FieldBytes<C>>::into(r).as_ref());
        let s = Scalar::from_be_bytes(Into::<FieldBytes<C>>::into(s).as_ref());
        let s_inv = Scalar::ONE.div_unsafe(s); // should IntMod have inv_unsafe?
        let u1 = &z * &s_inv;
        let u2 = &r * &s_inv;

        let G = Point::generator();
        let Q = Point::from_encoded_point::<C>(&self.0.to_encoded_point(false));
        let result = msm(&[u1, u2], &[G, Q]);

        // TODO: this only works when coord and scalar has same number of bytes
        let x_in_scalar = Scalar::from_le_bytes(result.x().as_le_bytes());
        let check = x_in_scalar - r;
        if check == Scalar::ZERO {
            Ok(())
        } else {
            Err(Error::new())
        }
    }
}
