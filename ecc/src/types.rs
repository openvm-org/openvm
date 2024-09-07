use afs_compiler::{
    ir::{BigUintVar, Builder, Config, MemIndex, MemVariable, Ptr, Variable},
    prelude::DslVariable,
};
use k256::{
    ecdsa::{Signature, VerifyingKey},
    sha2::digest::generic_array::GenericArray,
    EncodedPoint,
};
use num_bigint_dig::BigUint;
use zkhash::ark_ff::Zero;

/// EC point in Rust. **Unsafe** to assume (x, y) is a point on the curve.
#[derive(Clone, Debug)]
pub struct ECPoint {
    pub x: BigUint,
    pub y: BigUint,
}

/// EC point in eDSL. Safe to assume (x, y) is a point on the curve.
#[derive(DslVariable, Clone, Debug)]
pub struct ECPointVariable<C: Config> {
    pub(crate) x: BigUintVar<C>,
    pub(crate) y: BigUintVar<C>,
}

#[derive(Clone, Debug)]
pub struct ECDSASignature {
    pub r: BigUint,
    pub s: BigUint,
}

#[derive(DslVariable, Clone, Debug)]
pub struct ECDSASignatureVariable<C: Config> {
    pub r: BigUintVar<C>,
    pub s: BigUintVar<C>,
}

#[derive(Clone, Debug)]
pub struct ECDSAInput {
    pub pubkey: ECPoint,
    pub sig: ECDSASignature,
    pub msg_hash: BigUint,
}

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
