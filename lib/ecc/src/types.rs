use ax_circuit_primitives::bigint::utils::big_uint_to_num_limbs;
use axvm_native_compiler::{
    ir::{Array, BigUintVar, Builder, Config, MemIndex, MemVariable, Ptr, RVar, Var, Variable},
    prelude::DslVariable,
};
use k256::{
    ecdsa::{Signature, VerifyingKey},
    sha2::digest::generic_array::GenericArray,
    EncodedPoint,
};
use num_bigint_dig::BigUint;
use p3_field::{AbstractField, PrimeField64};
use zkhash::ark_ff::Zero;

/// EC point in Rust. **Unsafe** to assume (x, y) is a point on the curve.
#[derive(Clone, Debug)]
pub struct ECPoint {
    pub x: BigUint,
    pub y: BigUint,
}

impl ECPoint {
    // FIXME: coord_bits is the number of bits of the coordinate field. This should be in a config somewhere
    pub fn load_const<C: Config>(
        &self,
        builder: &mut Builder<C>,
        coord_bits: usize,
    ) -> ECPointVariable<C> {
        let limb_bits = builder.bigint_repr_size as usize;
        let num_limbs = (coord_bits + limb_bits - 1) / limb_bits;
        let array = builder.array(2 * num_limbs);

        let [x, y] = [&self.x, &self.y].map(|x| -> Vec<_> {
            big_uint_to_num_limbs(x, limb_bits, num_limbs)
                .into_iter()
                .map(C::N::from_canonical_usize)
                .collect()
        });
        for (i, &elem) in x.iter().chain(y.iter()).enumerate() {
            builder.set(&array, i, elem);
        }
        ECPointVariable { affine: array }
    }
}

/// EC point in eDSL. **Unsafe** to assume (x, y) is a point on the curve.
#[derive(DslVariable, Clone, Debug)]
pub struct ECPointVariable<C: Config> {
    /// Affine (x,y) as an array
    pub affine: Array<C, Var<C::N>>,
}

impl<C: Config> ECPointVariable<C> {
    // FIXME: coord_bits is the number of bits of the coordinate field. This should be in a config somewhere
    pub fn x(&self, builder: &mut Builder<C>, coord_bits: usize) -> BigUintVar<C> {
        let num_limbs = ((coord_bits as u32 + builder.bigint_repr_size - 1)
            / builder.bigint_repr_size) as usize;
        self.affine.slice(builder, 0, num_limbs)
    }
    pub fn y(&self, builder: &mut Builder<C>, coord_bits: usize) -> BigUintVar<C> {
        let num_limbs = ((coord_bits as u32 + builder.bigint_repr_size - 1)
            / builder.bigint_repr_size) as usize;
        self.affine.slice(builder, num_limbs, 2 * num_limbs)
    }
}

/// ECDSA signature in Rust. **Unsafe** to assume r, s is valid(in [1, n-1]).
#[derive(Clone, Debug)]
pub struct ECDSASignature {
    pub r: BigUint,
    pub s: BigUint,
}

/// ECDSA signature in eDSL. **Unsafe** to assume r, s is valid(in [1, n-1]).
#[derive(DslVariable, Clone, Debug)]
pub struct ECDSASignatureVariable<C: Config> {
    pub r: BigUintVar<C>,
    pub s: BigUintVar<C>,
}

/// ECDSA Input in Rust. **Unsafe** to assume validness.
#[derive(Clone, Debug)]
pub struct ECDSAInput {
    pub pubkey: ECPoint,
    pub sig: ECDSASignature,
    pub msg_hash: BigUint,
}

/// ECDSA Input in eDSL. **Unsafe** to assume validness.
#[derive(DslVariable, Clone, Debug)]
pub struct ECDSAInputVariable<C: Config> {
    pub pubkey: ECPointVariable<C>,
    pub sig: ECDSASignatureVariable<C>,
    pub msg_hash: BigUintVar<C>,
}

impl From<VerifyingKey> for ECPoint {
    fn from(value: VerifyingKey) -> Self {
        value.to_encoded_point(false).into()
    }
}

impl From<EncodedPoint> for ECPoint {
    fn from(value: EncodedPoint) -> Self {
        let coord_to_biguint = |opt_arr: Option<&GenericArray<u8, _>>| match opt_arr {
            Some(arr) => BigUint::from_bytes_be(arr.as_slice()),
            None => BigUint::zero(),
        };
        let x = coord_to_biguint(value.x());
        let y = coord_to_biguint(value.y());
        ECPoint { x, y }
    }
}

impl From<Signature> for ECDSASignature {
    fn from(value: Signature) -> Self {
        let (r, s) = value.split_bytes();
        ECDSASignature {
            r: BigUint::from_bytes_be(r.as_slice()),
            s: BigUint::from_bytes_be(s.as_slice()),
        }
    }
}

impl<C: Config> ECPointVariable<C>
where
    C::N: PrimeField64,
{
    // FIXME: only works for secp256k1 right now
    /// Return 1 if the point is valid. Otherwise, return 0.
    pub fn is_valid(&self, builder: &mut Builder<C>) -> Var<C::N> {
        let x = self.x(builder, 256);
        let y = self.y(builder, 256);
        let x_is_0 = builder.secp256k1_coord_is_zero(&x);
        let y_is_0 = builder.secp256k1_coord_is_zero(&y);
        let ret: Var<_> = builder.uninit();
        builder.if_eq(x_is_0 * y_is_0, RVar::one()).then_or_else(
            |builder| {
                builder.assign(&ret, RVar::one());
            },
            |builder| {
                let x2 = builder.secp256k1_coord_mul(&x, &x);
                let x3 = builder.secp256k1_coord_mul(&x2, &x);
                let c7 = builder.eval_biguint(7u64.into());
                let x3_plus_7 = builder.secp256k1_coord_add(&x3, &c7);
                let y2 = builder.secp256k1_coord_mul(&y, &y);
                let on_curve = builder.secp256k1_coord_eq(&y2, &x3_plus_7);
                builder.assign(&ret, on_curve);
            },
        );
        ret
    }
}

impl<C: Config> ECDSASignatureVariable<C>
where
    C::N: PrimeField64,
{
    /// Return 1 if the signature is valid. Otherwise, return 0.
    pub fn is_valid(&self, builder: &mut Builder<C>) -> Var<C::N> {
        let Self { r, s } = self;
        let r_is_0 = builder.secp256k1_scalar_is_zero(r);
        let s_is_0 = builder.secp256k1_scalar_is_zero(s);
        builder.eval((RVar::one() - r_is_0) * (RVar::one() - s_is_0))
    }
}

impl<C: Config> ECDSAInputVariable<C>
where
    C::N: PrimeField64,
{
    /// Return 1 if the input is valid. Otherwise, return 0.
    pub fn is_valid(&self, builder: &mut Builder<C>) -> Var<C::N> {
        let sig_is_valid = self.sig.is_valid(builder);
        let ret = builder.uninit();
        builder.if_eq(sig_is_valid, C::N::one()).then_or_else(
            |builder| {
                let pk_is_valid = self.pubkey.is_valid(builder);
                builder.assign(&ret, pk_is_valid);
            },
            |builder| {
                builder.assign(&ret, C::N::zero());
            },
        );
        ret
    }
}
