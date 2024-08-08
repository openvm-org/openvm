use core::mem::size_of;
use std::mem::transmute;

use afs_derive::AlignedBorrow;
use p3_keccak_air::KeccakCols as KeccakPermCols;
use p3_util::indices_arr;

use super::{
    KECCAK_CAPACITY_U16S, KECCAK_DIGEST_BYTES, KECCAK_RATE_BYTES, KECCAK_RATE_U16S,
    KECCAK_WIDTH_MINUS_DIGEST_U16S,
};

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakVmCols<T> {
    /// Columns for keccak-f permutation
    pub inner: KeccakPermCols<T>,
    /// Columns for sponge and padding
    pub sponge: KeccakSpongeCols<T>,
    /// Columns for opcode interface and operand memory access
    pub opcode: KeccakOpcodeCols<T>,
}

/// Columns specific to the KECCAK256 opcode.
/// The opcode instruction format is (a, b, len, d, e)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct KeccakOpcodeCols<T> {
    /// The starting timestamp to use for memory access in this row.
    /// A single row will do multiple memory accesses.
    pub start_timestamp: T,
    pub a: T,
    pub b: T,
    /// The remaining length of the unpadded input, in bytes.
    pub len: T,
    pub d: T,
    pub e: T,
    /// dst <- proj(word[a]_d)
    pub dst: T,
    /// src <- proj(word[b]_d)
    pub src: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct KeccakSpongeCols<T> {
    /// Only used on first row of a round to determine whether the preimage state should
    /// be reset to all 0s.
    pub is_new_start: T,

    /// The number of input bytes that have already been absorbed prior to this
    /// block.
    pub already_absorbed_bytes: T,

    /// Whether the current byte is a padding byte.
    ///
    /// If this row represents a full input block, this should contain all 0s.
    pub is_padding_byte: [T; KECCAK_RATE_BYTES],

    /// The block being absorbed, which may contain input bytes and padding
    /// bytes.
    pub block_bytes: [T; KECCAK_RATE_BYTES],

    /// For each of the first [KECCAK_RATE_U16S] `u16` limbs in the updated state,
    /// the most significant byte of the limb.
    pub updated_state_hi: [T; KECCAK_RATE_U16S],
}

impl<T> KeccakVmCols<T> {
    pub const fn remaining_len(&self) -> T {
        self.opcode.len
    }

    pub const fn is_new_start(&self) -> T {
        self.sponge.is_new_start
    }
}

pub const NUM_KECCAK_SPONGE_COLS: usize = size_of::<KeccakSpongeCols<u8>>();
pub(crate) const KECCAK_SPONGE_COL_MAP: KeccakSpongeCols<usize> = make_col_map();

const fn make_col_map() -> KeccakSpongeCols<usize> {
    let indices_arr = indices_arr::<NUM_KECCAK_SPONGE_COLS>();
    unsafe { transmute::<[usize; NUM_KECCAK_SPONGE_COLS], KeccakSpongeCols<usize>>(indices_arr) }
}

pub const NUM_KECCAK_VM_COLS: usize = size_of::<KeccakVmCols<u8>>();
