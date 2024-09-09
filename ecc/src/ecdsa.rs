use afs_compiler::ir::{Builder, Config, Var};
use elliptic_curve::sec1::ToEncodedPoint;
use p3_field::{AbstractField, PrimeField64};

use crate::{
    ec_mul::scalar_multiply_secp256k1,
    types::{ECDSAInputVariable, ECPoint, ECPointVariable},
};

/// Return 1 if the ECDSA verification succeeds. Otherwise, return 0.
/// **Assumption**: `input` is a valid ECDSA input.
/// Reference: https://en.wikipedia.org/wiki/Elliptic_Curve_Digital_Signature_Algorithm#Signature_verification_algorithm
///
/// **Caution:** This function does not perform input validation. This should be done separately via `input.is_valid`.
/// This is not done in this function since the input may share a public key so the input validation can be shared.
pub fn verify_ecdsa_secp256k1<C: Config>(
    builder: &mut Builder<C>,
    input: &ECDSAInputVariable<C>,
    window_bits: usize,
) -> Var<C::N>
where
    C::N: PrimeField64,
{
    let z = &input.msg_hash;
    // TODO: make sure input validation checks that r,s < modulus of scalar field
    let u1 = builder.secp256k1_scalar_div(z, &input.sig.s);
    let u2 = builder.secp256k1_scalar_div(&input.sig.r, &input.sig.s);
    // TODO: do we need to enforce u1, u2 are < modulus of scalar field?
    let generator = load_generator_secp256k1(builder);
    let u1_g = scalar_multiply_secp256k1(builder, &generator, u1, window_bits);
    let u2_qa = scalar_multiply_secp256k1(builder, &input.pubkey, u2, window_bits);
    let (x1, y1) = builder.ec_add(&(u1_g.x, u1_g.y), &(u2_qa.x, u2_qa.y));

    let ret = builder.uninit();

    let x1_is_0 = builder.secp256k1_coord_is_zero(&x1);
    let y1_is_0 = builder.secp256k1_coord_is_zero(&y1);
    builder.if_eq(x1_is_0 * y1_is_0, C::N::one()).then_or_else(
        |builder| {
            builder.assign(&ret, C::N::zero());
        },
        |builder| {
            let r_eq_x1 = builder.secp256k1_scalar_eq(&input.sig.r, &x1);
            builder.assign(&ret, r_eq_x1);
        },
    );
    ret
}

pub fn load_generator_secp256k1<C: Config>(builder: &mut Builder<C>) -> ECPointVariable<C>
where
    C::N: PrimeField64,
{
    let g = k256::AffinePoint::GENERATOR.to_encoded_point(false);
    let g: ECPoint = g.into();
    // Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    let x = builder.eval_biguint(g.x);
    // Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    let y = builder.eval_biguint(g.y);
    ECPointVariable { x, y }
}
