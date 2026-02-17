use std::array::from_fn;

use itertools::Itertools;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::{PrimeCharacteristicRing, PrimeField32};

pub const F_NUM_BYTES: usize = 4;
pub const COMMIT_NUM_BYTES: usize = DIGEST_SIZE * F_NUM_BYTES;
pub const OUTPUT_TOTAL_BYTES: usize = F_NUM_BYTES + COMMIT_NUM_BYTES;

pub fn byte_commit_to_f<F: PrimeCharacteristicRing, T: Into<F> + Clone>(
    byte_commit: &[T; COMMIT_NUM_BYTES],
) -> [F; DIGEST_SIZE] {
    byte_commit
        .chunks_exact(F_NUM_BYTES)
        .map(|chunk| bytes_to_f(chunk))
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

pub fn bytes_to_f<F: PrimeCharacteristicRing, T: Into<F> + Clone>(register: &[T]) -> F {
    assert_eq!(register.len(), F_NUM_BYTES);
    register.iter().enumerate().fold(F::ZERO, |acc, (i, limb)| {
        acc + (limb.clone().into() * F::from_usize(1 << (i * RV32_CELL_BITS)))
    })
}

pub fn combine_output<T>(
    output_commit: [T; COMMIT_NUM_BYTES],
    output_len: [T; F_NUM_BYTES],
) -> [T; OUTPUT_TOTAL_BYTES] {
    output_commit
        .into_iter()
        .chain(output_len)
        .collect_array()
        .unwrap()
}

pub fn split_output<T>(
    output: [T; OUTPUT_TOTAL_BYTES],
) -> ([T; COMMIT_NUM_BYTES], [T; F_NUM_BYTES]) {
    let mut it = output.into_iter();
    let commit = from_fn(|_| it.next().unwrap());
    let len = from_fn(|_| it.next().unwrap());
    (commit, len)
}
