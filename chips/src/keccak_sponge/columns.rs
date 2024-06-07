use core::mem::{size_of, transmute};

use afs_derive::AlignedBorrow;
use p3_util::indices_arr;

/// Total number of sponge bytes: number of rate bytes + number of capacity
/// bytes.
pub const KECCAK_WIDTH_BYTES: usize = 200;
/// Total number of 16-bit limbs in the sponge.
pub const KECCAK_WIDTH_U16S: usize = KECCAK_WIDTH_BYTES / 2;
/// Number of non-digest bytes.
pub const KECCAK_WIDTH_MINUS_DIGEST_U16S: usize = (KECCAK_WIDTH_BYTES - KECCAK_DIGEST_BYTES) / 2;
/// Number of rate bytes.
pub const KECCAK_RATE_BYTES: usize = 136;
/// Number of 16-bit rate limbs.
pub const KECCAK_RATE_U16S: usize = KECCAK_RATE_BYTES / 2;
/// Number of capacity bytes.
pub const KECCAK_CAPACITY_BYTES: usize = 64;
/// Number of 16-bit capacity limbs.
pub const KECCAK_CAPACITY_U16S: usize = KECCAK_CAPACITY_BYTES / 2;
/// Number of output digest bytes used during the squeezing phase.
pub const KECCAK_DIGEST_BYTES: usize = 32;
/// Number of 16-bit digest limbs.
pub const KECCAK_DIGEST_U16S: usize = KECCAK_DIGEST_BYTES / 2;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct KeccakSpongeCols<T> {
    pub timestamp: T,

    pub base_addr: T,

    /// 1 if this row represents a full input block, i.e. one in which each byte
    /// is an input byte, not a padding byte; 0 otherwise.
    pub is_full_input_block: T,

    /// The number of input bytes that have already been absorbed prior to this
    /// block.
    pub already_absorbed_bytes: T,

    /// Whether the current byte is a padding byte.
    ///
    /// If this row represents a full input block, this should contain all 0s.
    pub is_padding_byte: [T; KECCAK_RATE_BYTES],

    /// The initial rate part of the sponge, at the start of this step.
    pub original_rate_u16s: [T; KECCAK_RATE_U16S],

    /// The capacity part of the sponge, encoded as 16-bit chunks, at the start
    /// of this step.
    pub original_capacity_u16s: [T; KECCAK_CAPACITY_U16S],

    /// The block being absorbed, which may contain input bytes and/or padding
    /// bytes.
    pub block_bytes: [T; KECCAK_RATE_BYTES],

    /// The rate part of the sponge, encoded as 16-bit chunks, after the current
    /// block is xor'd in, but before the permutation is applied.
    pub xored_rate_u16s: [T; KECCAK_RATE_U16S],

    /// The entire state (rate + capacity) of the sponge, encoded as 16-bit
    /// chunks, after the permutation is applied, minus the first limbs
    /// where the digest is extracted from. Those missing limbs can be
    /// recomputed from their corresponding bytes stored in
    /// `updated_digest_state_bytes`.
    pub partial_updated_state_u16s: [T; KECCAK_WIDTH_MINUS_DIGEST_U16S],

    /// The first part of the state of the sponge, seen as bytes, after the
    /// permutation is applied. This also represents the output digest of
    /// the Keccak sponge during the squeezing phase.
    pub updated_digest_state_bytes: [T; KECCAK_DIGEST_BYTES],
}

pub const NUM_KECCAK_SPONGE_COLS: usize = size_of::<KeccakSpongeCols<u8>>();
pub(crate) const KECCAK_SPONGE_COL_MAP: KeccakSpongeCols<usize> = make_col_map();

const fn make_col_map() -> KeccakSpongeCols<usize> {
    let indices_arr = indices_arr::<NUM_KECCAK_SPONGE_COLS>();
    unsafe { transmute::<[usize; NUM_KECCAK_SPONGE_COLS], KeccakSpongeCols<usize>>(indices_arr) }
}
