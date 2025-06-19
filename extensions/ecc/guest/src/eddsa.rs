use openvm_algebra_guest::{IntMod, Reduce};
use openvm_sha2::sha256;

use crate::{edwards::TwistedEdwardsPoint, CyclicGroup, FromCompressed, IntrinsicCurve};

extern crate alloc;
use alloc::vec::Vec;

type Coordinate<C> = <<C as IntrinsicCurve>::Point as TwistedEdwardsPoint>::Coordinate;
type Scalar<C> = <C as IntrinsicCurve>::Scalar;
type Point<C> = <C as IntrinsicCurve>::Point;

#[repr(C)]
#[derive(Clone)]
pub struct VerifyingKey<C: IntrinsicCurve> {
    /// Affine point
    point: Point<C>,
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

    pub fn verify(&self, message: &[u8], sig: &[u8]) -> bool {
        let Some(sig) = Signature::<C>::from_bytes(sig) else {
            return false;
        };

        // TODO: replace with sha512
        let prehash = sha256(message);

        // h = SHA512(dom2(F, C) || R || A || PH(M))
        // RFC reference: https://datatracker.ietf.org/doc/html/rfc8032#section-5.1.7
        let mut sha_input = Vec::new();

        // dom2(F, C) domain separator
        // RFC reference: https://datatracker.ietf.org/doc/html/rfc8032#section-2
        // See definition of dom2 in the RFC. Note that the RFC refers to the prehash
        // version of Ed25519 as Ed25519ph, and the non-prehash version as Ed25519.
        sha_input.extend_from_slice(b"SigEd25519 no Ed25519 collisions");
        sha_input.extend_from_slice(&[1]); // phflag = 1

        // The RFC specifies optional "context" bytes that are shared between a signer and verifier.
        // We don't use any context bytes.
        // See: https://datatracker.ietf.org/doc/html/rfc8032#section-5.1
        sha_input.extend_from_slice(&[0]); // context len = 0

        sha_input.extend_from_slice(&encode_point::<C>(&sig.r));
        sha_input.extend_from_slice(&encode_point::<C>(&self.point));
        sha_input.extend_from_slice(&prehash);

        // TOOD: replace with sha512
        let h = sha256(&sha_input);

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
        res == <Point<C> as TwistedEdwardsPoint>::IDENTITY
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
    let rec_id = y_bytes[Coordinate::<C>::NUM_LIMBS - 1] & 0b10000000;
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
        y_bytes[Coordinate::<C>::NUM_LIMBS - 1] |= 0b10000000;
    }
    y_bytes
}
