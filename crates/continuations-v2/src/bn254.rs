use std::array::from_fn;

use num_bigint::BigUint;
use openvm_stark_sdk::p3_bn254::Bn254;
use p3_field::{PrimeCharacteristicRing, PrimeField, PrimeField32};
use stark_backend_v2::{DIGEST_SIZE, F};

pub const BN254_BYTES: usize = 32;

/// Wrapper for an array of big-endian bytes, representing an unsigned big integer. Each commit can
/// be converted to a Bn254 using the trivial identification as natural numbers or into a `u32`
/// digest by decomposing the big integer base-`F::MODULUS`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CommitBytes([u8; BN254_BYTES]);

impl CommitBytes {
    pub fn new(bytes: [u8; BN254_BYTES]) -> Self {
        Self(bytes)
    }

    pub fn as_slice(&self) -> &[u8; BN254_BYTES] {
        &self.0
    }

    pub fn reverse(&mut self) {
        self.0.reverse();
    }
}

impl From<[F; DIGEST_SIZE]> for CommitBytes {
    fn from(value: [F; DIGEST_SIZE]) -> Self {
        Self::from(babybear_digest_to_bn254(&value))
    }
}

impl From<[u32; DIGEST_SIZE]> for CommitBytes {
    fn from(value: [u32; DIGEST_SIZE]) -> Self {
        Self(u32_digest_to_bytes(&value))
    }
}

impl From<Bn254> for CommitBytes {
    fn from(value: Bn254) -> Self {
        Self(bn254_to_bytes(value))
    }
}

impl Into<[u32; DIGEST_SIZE]> for CommitBytes {
    fn into(self) -> [u32; DIGEST_SIZE] {
        bytes_to_u32_digest(&self.0)
    }
}

impl Into<Bn254> for CommitBytes {
    fn into(self) -> Bn254 {
        bytes_to_bn254(&self.0)
    }
}

fn babybear_digest_to_bn254(digest: &[F; DIGEST_SIZE]) -> Bn254 {
    let mut ret = Bn254::ZERO;
    let order = Bn254::from_u32(F::ORDER_U32);
    let mut base = Bn254::ONE;
    digest.iter().for_each(|&x| {
        ret += base * Bn254::from_u32(x.as_canonical_u32());
        base *= order;
    });
    ret
}

fn bytes_to_bn254(bytes: &[u8; BN254_BYTES]) -> Bn254 {
    let order = Bn254::from_u32(1 << 8);
    let mut ret = Bn254::ZERO;
    let mut base = Bn254::ONE;
    for byte in bytes.iter().rev() {
        ret += base * Bn254::from_u8(*byte);
        base *= order;
    }
    ret
}

fn bn254_to_bytes(bn254: Bn254) -> [u8; BN254_BYTES] {
    let mut ret = [0u8; BN254_BYTES];
    let bytes = bn254.as_canonical_biguint().to_bytes_be();
    let start = BN254_BYTES - bytes.len();
    ret[start..].copy_from_slice(&bytes);
    ret
}

fn bytes_to_u32_digest(bytes: &[u8; BN254_BYTES]) -> [u32; DIGEST_SIZE] {
    let mut bigint = BigUint::ZERO;
    for byte in bytes.iter() {
        bigint <<= 8;
        bigint += BigUint::from(*byte);
    }
    let order = F::ORDER_U32;
    from_fn(|_| {
        let bigint_digit = bigint.clone() % order;
        let digit = if bigint_digit == BigUint::ZERO {
            0u32
        } else {
            bigint_digit.to_u32_digits()[0]
        };
        bigint /= order;
        digit
    })
}

fn u32_digest_to_bytes(digest: &[u32; DIGEST_SIZE]) -> [u8; BN254_BYTES] {
    bn254_to_bytes(babybear_digest_to_bn254(&digest.map(F::from_u32)))
}
