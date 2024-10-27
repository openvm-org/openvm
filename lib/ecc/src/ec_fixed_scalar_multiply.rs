use axvm_native_compiler::ir::{Array, BigUintVar, Builder, Config, RVar, Var, NUM_LIMBS};
use ff::PrimeField;
use num_bigint_dig::BigUint;
use p3_field::{AbstractField, PrimeField64};
use snark_verifier_sdk::snark_verifier::halo2_base::utils::CurveAffineExt;

use crate::types::{ECPoint, ECPointVariable};

pub const BIGINT_MAX_BITS: usize = 256;

pub struct CachedPoints<C, P>
where
    C: Config,
    P: CurveAffineExt,
{
    pub original_point: P,
    pub window_bits: usize,
    /// This uses a vec of arrays because we always traverse/access the vect deterministically
    pub points: Vec<Array<C, ECPointVariable<C>>>,
}

impl<C, P> CachedPoints<C, P>
where
    C: Config,
    C::N: PrimeField64,
    P: CurveAffineExt,
    P::Base: PrimeField,
{
    pub fn new(builder: &mut Builder<C>, point: P, window_bits: usize, max_bits: usize) -> Self {
        let mut points = Vec::new();
        let mut increment = point;
        let num_windows = max_bits / window_bits;
        let window_len = (1usize << window_bits) - 1;
        for _ in 0..num_windows {
            let mut curr = increment;
            let cache_vec: Array<C, ECPointVariable<C>> = builder.dyn_array(window_len);
            for j in 0..window_len {
                let prev = curr;
                curr = (curr + increment).into();
                let prev = prev.into_coordinates();
                let var = ECPoint {
                    x: BigUint::from_bytes_le(prev.0.to_repr().as_ref()),
                    y: BigUint::from_bytes_le(prev.1.to_repr().as_ref()),
                }
                .load_const(builder, max_bits);
                builder.set(&cache_vec, j, var);
            }
            increment = curr;
            points.push(cache_vec);
        }
        CachedPoints {
            original_point: point,
            window_bits,
            points,
        }
    }
}

pub fn fixed_scalar_multiply_secp256k1<C, P>(
    builder: &mut Builder<C>,
    cached_points: &CachedPoints<C, P>,
    scalar: BigUintVar<C>,
) -> ECPointVariable<C>
where
    C: Config,
    C::N: PrimeField64,
    P: CurveAffineExt,
    P::Base: PrimeField,
{
    let window_bits = cached_points.window_bits;
    assert_eq!(BIGINT_MAX_BITS % window_bits, 0);
    // FIXME: configurable num limbs
    let result = builder.array(NUM_LIMBS * 2);
    for i in 0..2 * NUM_LIMBS {
        builder.set(&result, i, C::N::zero());
    }

    let bits = builder.num2bits_biguint(&scalar);
    for (i, cache_vec) in cached_points.points.iter().enumerate() {
        let window_sum: Var<C::N> = builder.uninit();
        builder.assign(&window_sum, RVar::zero());
        for j in 0..window_bits {
            let bit = builder.get(&bits, RVar::from(i * window_bits + window_bits - j - 1));
            builder.assign(&window_sum, window_sum + window_sum + bit);
        }
        builder.if_ne(window_sum, C::N::zero()).then(|builder| {
            builder.assign(&window_sum, window_sum - RVar::one());
            let point = builder.get(cache_vec, window_sum);
            let new_res = builder.secp256k1_add(result.clone(), point.affine);
            builder.assign(&result, new_res);
        });
    }
    ECPointVariable { affine: result }
}
