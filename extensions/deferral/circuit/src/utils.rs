use itertools::{fold, Itertools};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::{PrimeCharacteristicRing, PrimeField32};

pub const F_NUM_BYTES: usize = 4;
pub const COMMIT_NUM_BYTES: usize = DIGEST_SIZE * F_NUM_BYTES;

pub fn byte_commit_to_f<F: PrimeCharacteristicRing, T: Into<F> + Clone>(
    byte_commit: &[T; COMMIT_NUM_BYTES],
) -> [F; DIGEST_SIZE] {
    byte_commit
        .chunks_exact(F_NUM_BYTES)
        .map(|chunk| {
            fold(chunk.iter().enumerate(), F::ZERO, |acc, (i, byte)| {
                let b: F = byte.clone().into();
                acc + (F::from_usize(1 << i) * b)
            })
        })
        .collect_array()
        .unwrap()
}

pub fn f_commit_to_bytes<F: PrimeField32>(f_commit: &[F; DIGEST_SIZE]) -> [u8; COMMIT_NUM_BYTES] {
    f_commit
        .iter()
        .flat_map(|f| f.as_canonical_u32().to_le_bytes())
        .collect_array()
        .unwrap()
}
