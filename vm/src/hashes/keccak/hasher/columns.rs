use core::mem::size_of;
use std::mem::transmute;

use afs_derive::AlignedBorrow;
use p3_air::AirBuilder;
use p3_keccak_air::KeccakCols as KeccakPermCols;
use p3_util::indices_arr;

use super::{KECCAK_RATE_BYTES, KECCAK_RATE_U16S};

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
    /// True for all rows that are part of opcode execution.
    /// False on dummy rows only used to pad the height.
    pub is_enabled: T,
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
#[derive(Debug, AlignedBorrow)]
pub struct KeccakSpongeCols<T> {
    /// Only used on first row of a round to determine whether the preimage state should
    /// be reset to all 0s.
    pub is_new_start: T,

    /// Whether the current byte is a padding byte.
    ///
    /// If this row represents a full input block, this should contain all 0s.
    pub is_padding_byte: [T; KECCAK_RATE_BYTES],

    /// The block being absorbed, which may contain input bytes and padding
    /// bytes.
    pub block_bytes: [T; KECCAK_RATE_BYTES],

    /// For each of the first [KECCAK_RATE_U16S] `u16` limbs in the state,
    /// the most significant byte of the limb.
    /// Here `state` is the postimage state if last round and the preimage
    /// state if first round. It can be junk if not first or last round.
    pub state_hi: [T; KECCAK_RATE_U16S],
}

impl<T: Copy> KeccakVmCols<T> {
    pub const fn remaining_len(&self) -> T {
        self.opcode.len
    }

    pub const fn is_new_start(&self) -> T {
        self.sponge.is_new_start
    }

    pub fn postimage(&self, y: usize, x: usize, limb: usize) -> T {
        // WARNING: once plonky3 commit is updated this needs to be changed to y, x
        self.inner.a_prime_prime_prime(x, y, limb)
    }

    pub fn is_last_round(&self) -> T {
        *self.inner.step_flags.last().unwrap()
    }
}

impl<T: Copy> KeccakOpcodeCols<T> {
    pub fn assert_eq<AB: AirBuilder>(&self, builder: &mut AB, other: Self)
    where
        T: Into<AB::Expr>,
    {
        builder.assert_eq(self.is_enabled, other.is_enabled);
        builder.assert_eq(self.start_timestamp, other.start_timestamp);
        builder.assert_eq(self.a, other.a);
        builder.assert_eq(self.b, other.b);
        builder.assert_eq(self.len, other.len);
        builder.assert_eq(self.d, other.d);
        builder.assert_eq(self.e, other.e);
        builder.assert_eq(self.dst, other.dst);
        builder.assert_eq(self.src, other.src);
    }
}

pub const NUM_KECCAK_SPONGE_COLS: usize = size_of::<KeccakSpongeCols<u8>>();
pub(crate) const KECCAK_SPONGE_COL_MAP: KeccakSpongeCols<usize> = make_col_map();

const fn make_col_map() -> KeccakSpongeCols<usize> {
    let indices_arr = indices_arr::<NUM_KECCAK_SPONGE_COLS>();
    unsafe { transmute::<[usize; NUM_KECCAK_SPONGE_COLS], KeccakSpongeCols<usize>>(indices_arr) }
}

pub const NUM_KECCAK_VM_COLS: usize = size_of::<KeccakVmCols<u8>>();
