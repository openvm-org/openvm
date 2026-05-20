use std::{array::from_fn, mem::size_of};

use itertools::Itertools;
use openvm_circuit::arch::{PostflightError, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES, U16_CELL_SIZE};
pub use openvm_circuit_primitives::U16_BITS;
use openvm_instructions::riscv::RV64_BYTE_BITS;
use openvm_riscv_circuit::adapters::{byte_ptr_to_u16_ptr_value, rv64_u16_block_to_bytes};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::{PrimeCharacteristicRing, PrimeField32};

pub const F_NUM_BYTES: usize = size_of::<u32>();
pub const COMMIT_NUM_BYTES: usize = DIGEST_SIZE * F_NUM_BYTES;
pub const OUTPUT_LEN_NUM_BYTES: usize = size_of::<u64>();
pub const OUTPUT_TOTAL_BYTES: usize = OUTPUT_LEN_NUM_BYTES + COMMIT_NUM_BYTES;
pub const U16_MASK: u32 = (1u32 << U16_BITS) - 1;

/// Number of u16 cells per BabyBear field element.
pub const F_NUM_U16S: usize = F_NUM_BYTES / U16_CELL_SIZE;
/// `DIGEST_SIZE` field elements expressed as u16 cells.
pub const COMMIT_NUM_U16S: usize = DIGEST_SIZE * F_NUM_U16S;
/// OutputKey length field expressed as u16 cells, including the zero-padded upper 32 bits.
pub const OUTPUT_LEN_NUM_U16S: usize = OUTPUT_LEN_NUM_BYTES / U16_CELL_SIZE;
/// Full output key `[output_commit || output_len]` expressed as u16 cells.
pub const OUTPUT_TOTAL_NUM_U16S: usize = COMMIT_NUM_U16S + OUTPUT_LEN_NUM_U16S;

/// Guest bytes absorbed by Poseidon2 per deferral output row.
pub const SPONGE_BYTES_PER_ROW: usize = U16_CELL_SIZE * DIGEST_SIZE;
/// Number of memory-bus messages used to write one sponge row to byte-addressed memory.
pub const SPONGE_ROW_MEMORY_OPS: usize = num_byte_memory_ops(SPONGE_BYTES_PER_ROW);
/// Number of memory-bus messages for a `DIGEST_SIZE`-cell DEFERRAL_AS chunk.
pub const DIGEST_F_MEMORY_OPS: usize = num_f_memory_ops(DIGEST_SIZE);
pub const COMMIT_MEMORY_OPS: usize = num_byte_memory_ops(COMMIT_NUM_BYTES);
pub const OUTPUT_TOTAL_MEMORY_OPS: usize = num_f_memory_ops(OUTPUT_TOTAL_NUM_U16S);

pub(crate) fn checked_u16_pointer(
    byte_pointer: u32,
    name: &'static str,
) -> Result<u32, PostflightError> {
    if byte_pointer & 1 != 0 {
        return Err(PostflightError::new(format!(
            "{name} is not two-byte aligned"
        )));
    }
    Ok(byte_ptr_to_u16_ptr_value(byte_pointer))
}

pub(crate) fn logged_u32_pointer(
    value: [u16; BLOCK_FE_WIDTH],
    name: &'static str,
) -> Result<u32, PostflightError> {
    let bytes = rv64_u16_block_to_bytes(value);
    if bytes[4..] != [0; 4] {
        return Err(PostflightError::new(format!(
            "{name} has nonzero upper bits"
        )));
    }
    Ok(u32::from_le_bytes(bytes[..4].try_into().unwrap()))
}

pub(crate) fn require_block_alignment(
    pointer: u32,
    name: &'static str,
) -> Result<(), PostflightError> {
    if !pointer.is_multiple_of(MEMORY_BLOCK_BYTES as u32) {
        return Err(PostflightError::new(format!(
            "{name} is not memory-block aligned"
        )));
    }
    Ok(())
}

pub(crate) fn checked_pointer_offset(
    pointer: u32,
    offset: usize,
    message: &'static str,
) -> Result<u32, PostflightError> {
    pointer
        .checked_add(
            u32::try_from(offset).map_err(|_| PostflightError::new("memory offset exceeds u32"))?,
        )
        .ok_or_else(|| PostflightError::new(message))
}

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

/// Split heap byte data into memory-bus chunks.
pub fn split_byte_memory_ops<T, const TOTAL_BYTES: usize, const NUM_OPS: usize>(
    data: [T; TOTAL_BYTES],
) -> [[T; MEMORY_BLOCK_BYTES]; NUM_OPS] {
    const {
        assert!(
            TOTAL_BYTES == NUM_OPS * MEMORY_BLOCK_BYTES,
            "TOTAL_BYTES must equal NUM_OPS * MEMORY_BLOCK_BYTES"
        )
    };
    let mut it = data.into_iter();
    from_fn(|_| from_fn(|_| it.next().unwrap()))
}

/// Split cell-shaped memory data into memory-bus chunks. Callers choose the address stride for
/// the address space they access.
pub fn split_cell_memory_ops<T, const TOTAL_CELLS: usize, const NUM_OPS: usize>(
    data: [T; TOTAL_CELLS],
) -> [[T; BLOCK_FE_WIDTH]; NUM_OPS] {
    const {
        assert!(
            TOTAL_CELLS == NUM_OPS * BLOCK_FE_WIDTH,
            "TOTAL_CELLS must equal NUM_OPS * BLOCK_FE_WIDTH"
        )
    };
    let mut it = data.into_iter();
    from_fn(|_| from_fn(|_| it.next().unwrap()))
}

pub fn join_byte_memory_ops<T, const TOTAL_BYTES: usize, const NUM_OPS: usize>(
    chunks: [[T; MEMORY_BLOCK_BYTES]; NUM_OPS],
) -> [T; TOTAL_BYTES] {
    const {
        assert!(
            TOTAL_BYTES == NUM_OPS * MEMORY_BLOCK_BYTES,
            "TOTAL_BYTES must equal NUM_OPS * MEMORY_BLOCK_BYTES"
        )
    };
    chunks.into_iter().flatten().collect_array().unwrap()
}

pub fn byte_memory_op_chunk<T: Clone>(data: &[T], chunk_idx: usize) -> [T; MEMORY_BLOCK_BYTES] {
    debug_assert!(data.len().is_multiple_of(MEMORY_BLOCK_BYTES));
    let start = chunk_idx * MEMORY_BLOCK_BYTES;
    debug_assert!(start + MEMORY_BLOCK_BYTES <= data.len());
    from_fn(|i| data[start + i].clone())
}

/// Split DEFERRAL_AS cell data into memory-bus chunks.
pub fn split_f_memory_ops<T, const TOTAL_CELLS: usize, const NUM_OPS: usize>(
    data: [T; TOTAL_CELLS],
) -> [[T; BLOCK_FE_WIDTH]; NUM_OPS] {
    const {
        assert!(
            TOTAL_CELLS == NUM_OPS * BLOCK_FE_WIDTH,
            "TOTAL_CELLS must equal NUM_OPS * BLOCK_FE_WIDTH"
        )
    };
    let mut it = data.into_iter();
    from_fn(|_| from_fn(|_| it.next().unwrap()))
}

pub fn join_f_memory_ops<T, const TOTAL_CELLS: usize, const NUM_OPS: usize>(
    chunks: [[T; BLOCK_FE_WIDTH]; NUM_OPS],
) -> [T; TOTAL_CELLS] {
    const {
        assert!(
            TOTAL_CELLS == NUM_OPS * BLOCK_FE_WIDTH,
            "TOTAL_CELLS must equal NUM_OPS * BLOCK_FE_WIDTH"
        )
    };
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

/// Compose a commit's little-endian u16 cells into field elements.
pub fn u16_commit_to_f<F: PrimeCharacteristicRing, T: Into<F> + Clone>(
    u16_commit: &[T],
) -> [F; DIGEST_SIZE] {
    assert_eq!(u16_commit.len(), COMMIT_NUM_U16S);
    u16_commit
        .chunks_exact(F_NUM_U16S)
        .map(u16s_to_f)
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

/// Pack little-endian byte pairs into u16 cells.
pub fn le_bytes_to_u16_array<const N: usize>(bytes: &[u8]) -> [u16; N] {
    assert_eq!(bytes.len(), U16_CELL_SIZE * N);
    from_fn(|i| {
        let offset = U16_CELL_SIZE * i;
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    })
}

/// Pack little-endian byte pairs into field-valued u16 cells.
pub fn le_bytes_to_u16_cells<F: PrimeCharacteristicRing, const N: usize>(bytes: &[u8]) -> [F; N] {
    le_bytes_to_u16_array(bytes).map(F::from_u16)
}

/// Decompose a field-valued commit into little-endian u16 cells.
pub fn f_commit_to_u16s<F: PrimeField32>(f_commit: &[F; DIGEST_SIZE]) -> [u16; COMMIT_NUM_U16S] {
    f_commit
        .iter()
        .flat_map(|f| le_bytes_to_u16_array::<F_NUM_U16S>(&f.as_canonical_u32().to_le_bytes()))
        .collect_array()
        .unwrap()
}

pub fn bytes_to_f<F: PrimeCharacteristicRing, T: Into<F> + Clone>(register: &[T]) -> F {
    assert_eq!(register.len(), F_NUM_BYTES);
    register.iter().enumerate().fold(F::ZERO, |acc, (i, limb)| {
        acc + (limb.clone().into() * F::from_usize(1 << (i * RV64_BYTE_BITS)))
    })
}

/// Compose little-endian u16 limbs into one field element.
pub fn u16s_to_f<F: PrimeCharacteristicRing, T: Into<F> + Clone>(register: &[T]) -> F {
    assert_eq!(register.len(), F_NUM_U16S);
    register.iter().enumerate().fold(F::ZERO, |acc, (i, limb)| {
        acc + (limb.clone().into() * F::from_usize(1 << (i * U16_BITS)))
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

pub fn combine_output_cells<T>(
    output_commit: impl IntoIterator<Item = T>,
    output_len: [T; OUTPUT_LEN_NUM_U16S],
) -> [T; OUTPUT_TOTAL_NUM_U16S] {
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
