use std::array::from_fn;

use itertools::Itertools;
use openvm_circuit::arch::{BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES};
use openvm_instructions::riscv::RV64_CELL_BITS;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::{PrimeCharacteristicRing, PrimeField32};

pub const F_NUM_BYTES: usize = 4;
pub const COMMIT_NUM_BYTES: usize = DIGEST_SIZE * F_NUM_BYTES;
pub const OUTPUT_LEN_NUM_BYTES: usize = 8;
pub const OUTPUT_TOTAL_BYTES: usize = OUTPUT_LEN_NUM_BYTES + COMMIT_NUM_BYTES;
/// Number of memory bus messages to read/write a `DIGEST_SIZE`-byte chunk from a
/// byte-addressed AS (RV64_MEMORY_AS). Each bus access covers `MEMORY_BLOCK_BYTES`
/// bytes (one guest-visible block).
pub const DIGEST_BYTE_MEMORY_OPS: usize = num_byte_memory_ops(DIGEST_SIZE);
/// Number of memory bus messages to read/write a `DIGEST_SIZE`-cell chunk from the
/// F-celled DEFERRAL_AS. Each bus access covers `BLOCK_FE_WIDTH` cells. With
/// `BLOCK_FE_WIDTH = MEMORY_BLOCK_BYTES = 8` today this equals
/// `DIGEST_BYTE_MEMORY_OPS`; after the Stage 1.6 flip `BLOCK_FE_WIDTH = 4` and the
/// count doubles (deferral chip emits two F bus messages per logical 8-F op).
pub const DIGEST_F_MEMORY_OPS: usize = num_f_memory_ops(DIGEST_SIZE);
pub const COMMIT_MEMORY_OPS: usize = num_byte_memory_ops(COMMIT_NUM_BYTES);
pub const OUTPUT_TOTAL_MEMORY_OPS: usize = num_byte_memory_ops(OUTPUT_TOTAL_BYTES);

#[inline(always)]
pub const fn num_byte_memory_ops(total_bytes: usize) -> usize {
    assert!(total_bytes.is_multiple_of(MEMORY_BLOCK_BYTES));
    total_bytes / MEMORY_BLOCK_BYTES
}

#[inline(always)]
pub const fn num_f_memory_ops(total_cells: usize) -> usize {
    assert!(total_cells.is_multiple_of(BLOCK_FE_WIDTH));
    total_cells / BLOCK_FE_WIDTH
}

/// Split `TOTAL_BYTES` bytes of byte-addressed-AS data into `NUM_OPS` chunks of
/// `MEMORY_BLOCK_BYTES` bytes each. Used for RV64_MEMORY_AS accesses.
pub fn split_byte_memory_ops<T, const TOTAL_BYTES: usize, const NUM_OPS: usize>(
    data: [T; TOTAL_BYTES],
) -> [[T; MEMORY_BLOCK_BYTES]; NUM_OPS] {
    assert_eq!(TOTAL_BYTES, NUM_OPS * MEMORY_BLOCK_BYTES);
    let mut it = data.into_iter();
    from_fn(|_| from_fn(|_| it.next().unwrap()))
}

pub fn join_byte_memory_ops<T, const TOTAL_BYTES: usize, const NUM_OPS: usize>(
    chunks: [[T; MEMORY_BLOCK_BYTES]; NUM_OPS],
) -> [T; TOTAL_BYTES] {
    assert_eq!(TOTAL_BYTES, NUM_OPS * MEMORY_BLOCK_BYTES);
    chunks.into_iter().flatten().collect_array().unwrap()
}

pub fn byte_memory_op_chunk<T: Clone>(data: &[T], chunk_idx: usize) -> [T; MEMORY_BLOCK_BYTES] {
    debug_assert!(data.len().is_multiple_of(MEMORY_BLOCK_BYTES));
    let start = chunk_idx * MEMORY_BLOCK_BYTES;
    debug_assert!(start + MEMORY_BLOCK_BYTES <= data.len());
    from_fn(|i| data[start + i].clone())
}

/// Split `TOTAL_CELLS` F cells of DEFERRAL_AS data into `NUM_OPS` chunks of
/// `BLOCK_FE_WIDTH` cells each. Used for DEFERRAL_AS accesses.
pub fn split_f_memory_ops<T, const TOTAL_CELLS: usize, const NUM_OPS: usize>(
    data: [T; TOTAL_CELLS],
) -> [[T; BLOCK_FE_WIDTH]; NUM_OPS] {
    assert_eq!(TOTAL_CELLS, NUM_OPS * BLOCK_FE_WIDTH);
    let mut it = data.into_iter();
    from_fn(|_| from_fn(|_| it.next().unwrap()))
}

pub fn join_f_memory_ops<T, const TOTAL_CELLS: usize, const NUM_OPS: usize>(
    chunks: [[T; BLOCK_FE_WIDTH]; NUM_OPS],
) -> [T; TOTAL_CELLS] {
    assert_eq!(TOTAL_CELLS, NUM_OPS * BLOCK_FE_WIDTH);
    chunks.into_iter().flatten().collect_array().unwrap()
}

pub fn f_memory_op_chunk<T: Clone>(data: &[T], chunk_idx: usize) -> [T; BLOCK_FE_WIDTH] {
    debug_assert!(data.len().is_multiple_of(BLOCK_FE_WIDTH));
    let start = chunk_idx * BLOCK_FE_WIDTH;
    debug_assert!(start + BLOCK_FE_WIDTH <= data.len());
    from_fn(|i| data[start + i].clone())
}

pub fn byte_commit_to_f<F: PrimeCharacteristicRing, T: Into<F> + Clone>(
    byte_commit: &[T],
) -> [F; DIGEST_SIZE] {
    assert_eq!(byte_commit.len(), COMMIT_NUM_BYTES);
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
        acc + (limb.clone().into() * F::from_usize(1 << (i * RV64_CELL_BITS)))
    })
}

pub fn combine_output<T>(
    output_commit: impl IntoIterator<Item = T>,
    output_len: [T; OUTPUT_LEN_NUM_BYTES],
) -> [T; OUTPUT_TOTAL_BYTES] {
    output_commit
        .into_iter()
        .chain(output_len)
        .collect_array()
        .unwrap()
}

pub fn split_output<T>(
    output: [T; OUTPUT_TOTAL_BYTES],
) -> ([T; COMMIT_NUM_BYTES], [T; OUTPUT_LEN_NUM_BYTES]) {
    let mut it = output.into_iter();
    let commit = from_fn(|_| it.next().unwrap());
    let len = from_fn(|_| it.next().unwrap());
    (commit, len)
}
