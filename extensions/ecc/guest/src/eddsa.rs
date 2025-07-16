// Implementation of the EdDSA signature verification algorithm.
// The code is generic over the twisted Edwards curve, but currently only instantiated with Ed25519.
// The implementation is based on the RFC: https://datatracker.ietf.org/doc/html/rfc8032
// We support both the prehash variant (Ed25519ph) and the non-prehash variant (Ed25519).
// Note: our implementation is not intended to be safe against timing attacks.

extern crate alloc;
use alloc::vec::Vec;

use openvm_sha2::sha512;

use crate::{
    algebra::{IntMod, Reduce},
    edwards::TwistedEdwardsPoint,
    CyclicGroup, FromCompressed, IntrinsicCurve,
};

type Coordinate<C> = <<C as IntrinsicCurve>::Point as TwistedEdwardsPoint>::Coordinate;
type Scalar<C> = <C as IntrinsicCurve>::Scalar;
type Point<C> = <C as IntrinsicCurve>::Point;

#[repr(C)]
#[derive(Clone)]
pub struct VerifyingKey<C: IntrinsicCurve> {
    /// Affine point
    point: Point<C>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationError {
    InvalidSignature,
    InvalidContext,
    FailedToVerify,
}

impl<C: IntrinsicCurve> VerifyingKey<C>
where
    Point<C>: TwistedEdwardsPoint + FromCompressed<Coordinate<C>> + CyclicGroup,
    Coordinate<C>: IntMod,
    C::Scalar: IntMod + Reduce,
{
    /// Assumes the point is encoded as in https://datatracker.ietf.org/doc/html/rfc8032#section-5.1.2
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Coordinate::<C>::NUM_LIMBS {
            return None;
        }
        Some(Self {
            point: decode_point::<C>(bytes)?,
        })
    }

    pub fn verify(&self, message: &[u8], sig: &[u8]) -> Result<(), VerificationError> {
        self.verify_prehashed(message, sig, &[])
    }

    /// The verify function for the prehash variant of Ed25519.
    /// message should be the message to be verified, before the prehash is applied.
    /// context is the optional context bytes that are shared between a signer and verifier, as per
    /// the Ed25519ph specification. If no context is provided, the empty slice will be used.
    /// The context can be up to 255 bytes.
    pub fn verify_ph(
        &self,
        message: &[u8],
        context: Option<&[u8]>,
        sig: &[u8],
    ) -> Result<(), VerificationError> {
        let prehash = sha512(message);

        // dom2(F, C) domain separator
        // RFC reference: https://datatracker.ietf.org/doc/html/rfc8032#section-2
        // See definition of dom2 in the RFC. Note that the RFC refers to the prehash
        // version of Ed25519 as Ed25519ph, and the non-prehash version as Ed25519.
        let mut dom2 = Vec::new();
        dom2.extend_from_slice(b"SigEd25519 no Ed25519 collisions");
        dom2.push(1); // phflag = 1

        // The RFC specifies optional "context" bytes that are shared between a signer and verifier.
        // See: https://datatracker.ietf.org/doc/html/rfc8032#section-5.1
        if let Some(context) = context {
            if context.len() > 255 {
                return Err(VerificationError::InvalidContext);
            }
            dom2.push(context.len() as u8);
            dom2.extend_from_slice(context);
        } else {
            dom2.push(0); // context len = 0
        }

        self.verify_prehashed(&prehash, sig, &dom2)
    }

    // Shared verify function for both the prehash and non-prehash variants of Ed25519.
    // prehash is either SHA512(message) or message, for Ed25519ph and Ed25519 respectively.
    // dom2 is the domain separator for the Ed25519ph and Ed25519 variants. It should be empty for
    // Ed25519.
    // See RFC reference: https://datatracker.ietf.org/doc/html/rfc8032#section-2
    fn verify_prehashed(
        &self,
        prehash: &[u8],
        sig: &[u8],
        dom2: &[u8],
    ) -> Result<(), VerificationError> {
        let Some(sig) = Signature::<C>::from_bytes(sig) else {
            return Err(VerificationError::InvalidSignature);
        };

        // h = SHA512(dom2(F, C) || R || A || PH(M))
        // RFC reference: https://datatracker.ietf.org/doc/html/rfc8032#section-5.1.7
        let mut sha_input = Vec::new();

        sha_input.extend_from_slice(dom2);

        sha_input.extend_from_slice(&encode_point::<C>(&sig.r));
        sha_input.extend_from_slice(&encode_point::<C>(&self.point));
        sha_input.extend_from_slice(prehash);

        let h = sha512(&sha_input);

        let h = C::Scalar::reduce_le_bytes(&h);

        // assert s * B = R + h * A
        // <=> R + h * A - s * B = 0
        // <=> [1, h, s] * [R, A, -B] = 0
        let res = C::msm(
            &[C::Scalar::ONE, h, sig.s],
            &[
                sig.r,
                self.point.clone(),
                <Point<C> as CyclicGroup>::NEG_GENERATOR,
            ],
        );
        if res == <Point<C> as TwistedEdwardsPoint>::IDENTITY {
            Ok(())
        } else {
            Err(VerificationError::FailedToVerify)
        }
    }
}

// Internal struct used for decoding the signature from bytes
struct Signature<C: IntrinsicCurve> {
    r: C::Point,
    s: C::Scalar,
}

impl<C: IntrinsicCurve> Signature<C>
where
    C::Point: TwistedEdwardsPoint + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
    C::Scalar: IntMod,
{
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Coordinate::<C>::NUM_LIMBS + Scalar::<C>::NUM_LIMBS {
            return None;
        }
        // from_le_bytes checks that s is reduced
        let s = Scalar::<C>::from_le_bytes(&bytes[Coordinate::<C>::NUM_LIMBS..])?;
        Some(Self {
            r: decode_point::<C>(&bytes[..Coordinate::<C>::NUM_LIMBS])?,
            s,
        })
    }
}

/// RFC reference: https://datatracker.ietf.org/doc/html/rfc8032#section-5.1.3
/// We require that the most significant bit in the little-endian encoding of
/// elements of the coordinate field is always 0, because we pack the parity
/// of the x-coordinate there.
fn decode_point<C: IntrinsicCurve>(bytes: &[u8]) -> Option<Point<C>>
where
    Point<C>: TwistedEdwardsPoint + FromCompressed<Coordinate<C>>,
    Coordinate<C>: IntMod,
{
    if bytes.len() != Coordinate::<C>::NUM_LIMBS {
        return None;
    }
    let mut y_bytes = bytes.to_vec();
    // most significant bit stores the parity of the x-coordinate
    let rec_id = (y_bytes[Coordinate::<C>::NUM_LIMBS - 1] & 0b10000000) >> 7;
    y_bytes[Coordinate::<C>::NUM_LIMBS - 1] &= 0b01111111;
    // from_le_bytes checks that y is reduced
    let y = Coordinate::<C>::from_le_bytes(&y_bytes)?;
    Point::<C>::decompress(y, &rec_id)
}

/// RFC reference: https://datatracker.ietf.org/doc/html/rfc8032#section-5.1.2
/// We require that the most significant bit in the little-endian encoding of
/// elements of the coordinate field is always 0, because we pack the parity
/// of the x-coordinate there.
fn encode_point<C: IntrinsicCurve>(p: &Point<C>) -> Vec<u8>
where
    Point<C>: TwistedEdwardsPoint,
    Coordinate<C>: IntMod,
{
    let mut y_bytes = p.y().as_le_bytes().to_vec();
    if p.x().as_le_bytes()[0] & 1u8 == 1 {
        // We pack the parity of the x-coordinate in the most significant bit of the last byte, as
        // per the Ed25519 spec, so the Coordinate<C> type must have enough limbs so that the most
        // significant bit of the last byte is always 0.
        debug_assert!(y_bytes[Coordinate::<C>::NUM_LIMBS - 1] & 0b10000000 == 0);
        y_bytes[Coordinate::<C>::NUM_LIMBS - 1] |= 0b10000000;
    }
    y_bytes
}
