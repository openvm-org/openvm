use afs_compiler::ir::{Builder, Config, Var};
use p3_field::{AbstractField, PrimeField64};
use snark_verifier_sdk::snark_verifier::halo2_base::halo2_proofs::halo2curves::secp256k1::Secp256k1Affine;

use crate::{
    ec_fixed_scalar_multiply::{fixed_scalar_multiply_secp256k1, CachedPoints},
    ec_mul::scalar_multiply_secp256k1,
    types::{ECDSAInputVariable, ECPointVariable},
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
    let cached_points = load_generator_secp256k1(builder);
    let u1_g = fixed_scalar_multiply_secp256k1(builder, &cached_points, u1);
    let u2_qa = scalar_multiply_secp256k1(builder, &input.pubkey, u2, window_bits);
    let sum_affine = builder.secp256k1_add(u1_g.affine, u2_qa.affine);
    let sum = ECPointVariable { affine: sum_affine };
    let x1 = sum.x(builder, 256);
    let y1 = sum.y(builder, 256);

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

pub fn load_generator_secp256k1<C: Config>(
    builder: &mut Builder<C>,
) -> CachedPoints<C, Secp256k1Affine>
where
    C::N: PrimeField64,
{
    let g = Secp256k1Affine::generator();
    // let g: ECPoint = g.into();
    // Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    // Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    CachedPoints::new(builder, g, 4, 256)
}
