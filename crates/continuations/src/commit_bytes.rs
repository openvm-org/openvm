use std::array::from_fn;

use num_bigint::BigUint;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use openvm_verify_stark_host::pvs::VkCommit;
use p3_field::{PrimeCharacteristicRing, PrimeField32};

pub const COMMIT_NUM_BYTES: usize = 32;

/// Wrapper for an array of big-endian bytes, representing an unsigned big integer. Each commit can
/// be converted to a Bn254 using the trivial identification as natural numbers or into a `u32`
/// digest by decomposing the big integer base-`F::MODULUS`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CommitBytes([u8; COMMIT_NUM_BYTES]);

impl CommitBytes {
    pub fn new(bytes: [u8; COMMIT_NUM_BYTES]) -> Self {
        Self(bytes)
    }

    pub fn as_slice(&self) -> &[u8; COMMIT_NUM_BYTES] {
        &self.0
    }

    pub fn reverse(&mut self) {
        self.0.reverse();
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VkCommitBytes {
    pub cached_commit: CommitBytes,
    pub pre_hash: CommitBytes,
}

impl<F: PrimeCharacteristicRing> From<VkCommitBytes> for VkCommit<F> {
    fn from(value: VkCommitBytes) -> Self {
        VkCommit {
            cached_commit: value.cached_commit.into(),
            vk_pre_hash: value.pre_hash.into(),
        }
    }
}

impl From<[F; DIGEST_SIZE]> for CommitBytes {
    fn from(value: [F; DIGEST_SIZE]) -> Self {
        Self(u32_digest_to_bytes(&value.map(|x| x.as_canonical_u32())))
    }
}

impl From<[u32; DIGEST_SIZE]> for CommitBytes {
    fn from(value: [u32; DIGEST_SIZE]) -> Self {
        Self(u32_digest_to_bytes(&value))
    }
}

impl<F: PrimeCharacteristicRing> From<CommitBytes> for [F; DIGEST_SIZE] {
    fn from(value: CommitBytes) -> Self {
        bytes_to_u32_digest(&value.0).map(F::from_u32)
    }
}

fn bytes_to_biguint(bytes: &[u8; COMMIT_NUM_BYTES]) -> BigUint {
    let mut bigint = BigUint::ZERO;
    for byte in bytes.iter() {
        bigint <<= 8;
        bigint += BigUint::from(*byte);
    }
    bigint
}

fn biguint_to_u32_digest(mut bigint: BigUint) -> [u32; DIGEST_SIZE] {
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

fn u32_digest_to_biguint(digest: &[u32; DIGEST_SIZE]) -> BigUint {
    let mut bigint = BigUint::ZERO;
    let mut base = BigUint::from(1u32);
    let order = BigUint::from(F::ORDER_U32);
    for digit in digest {
        bigint += &base * BigUint::from(*digit);
        base *= &order;
    }
    bigint
}

fn bytes_to_u32_digest(bytes: &[u8; COMMIT_NUM_BYTES]) -> [u32; DIGEST_SIZE] {
    biguint_to_u32_digest(bytes_to_biguint(bytes))
}

fn u32_digest_to_bytes(digest: &[u32; DIGEST_SIZE]) -> [u8; COMMIT_NUM_BYTES] {
    let mut ret = [0u8; COMMIT_NUM_BYTES];
    let bytes = u32_digest_to_biguint(digest).to_bytes_be();
    let start = COMMIT_NUM_BYTES - bytes.len();
    ret[start..].copy_from_slice(&bytes);
    ret
}

#[cfg(feature = "root-prover")]
mod bn254 {
    use p3_bn254::Bn254;
    use p3_field::PrimeField;

    use super::*;

    impl From<Bn254> for CommitBytes {
        fn from(value: Bn254) -> Self {
            Self(bn254_to_bytes(value))
        }
    }

    impl From<[Bn254; 1]> for CommitBytes {
        fn from(value: [Bn254; 1]) -> Self {
            CommitBytes::from(value[0])
        }
    }

    impl From<CommitBytes> for Bn254 {
        fn from(value: CommitBytes) -> Self {
            bytes_to_bn254(&value.0)
        }
    }

    fn bytes_to_bn254(bytes: &[u8; COMMIT_NUM_BYTES]) -> Bn254 {
        let order = Bn254::from_u32(1 << 8);
        let mut ret = Bn254::ZERO;
        let mut base = Bn254::ONE;
        for byte in bytes.iter().rev() {
            ret += base * Bn254::from_u8(*byte);
            base *= order;
        }
        ret
    }

    fn bn254_to_bytes(bn254: Bn254) -> [u8; COMMIT_NUM_BYTES] {
        let mut ret = [0u8; COMMIT_NUM_BYTES];
        let bytes = bn254.as_canonical_biguint().to_bytes_be();
        let start = COMMIT_NUM_BYTES - bytes.len();
        ret[start..].copy_from_slice(&bytes);
        ret
    }
}
