use core::mem::size_of;

use ax_circuit_derive::AlignedBorrow;
use p3_air::AirBuilder;
use p3_keccak_air::KeccakCols as KeccakPermCols;

use super::{
    KECCAK_ABSORB_READS, KECCAK_DIGEST_WRITES, KECCAK_EXECUTION_READS, KECCAK_RATE_BYTES,
    KECCAK_RATE_U16S, KECCAK_WORD_SIZE,
};
use crate::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakVmCols<T> {
    /// Columns for keccak-f permutation
    pub inner: KeccakPermCols<T>,
    /// Columns for sponge and padding
    pub sponge: KeccakSpongeCols<T>,
    /// Columns for opcode interface and operand memory access
    pub opcode: KeccakOpcodeCols<T>,
    /// Auxiliary columns for offline memory checking
    pub mem_oc: KeccakMemoryCols<T>,
}

/// Columns specific to the KECCAK256 opcode.
/// The opcode instruction format is (a, b, len, d, e, f)
#[allow(clippy::too_many_arguments)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct KeccakOpcodeCols<T> {
    /// Program counter
    pub pc: T,
    /// True for all rows that are part of opcode execution.
    /// False on dummy rows only used to pad the height.
    pub is_enabled: T,
    /// Is enabled and first round of block. Used to lower constraint degree.
    /// is_enabled * inner.step_flags[0]
    pub is_enabled_first_round: T,
    /// The starting timestamp to use for memory access in this row.
    /// A single row will do multiple memory accesses.
    pub start_timestamp: T,
    // Operands:
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub f: T,
    // Memory values
    /// dst <- [a]_d
    pub dst: T,
    /// src <- [b]_d
    pub src: T,
    /// The remaining length of the unpadded input, in bytes.
    /// If this row is receiving from opcode bus, then
    /// len <- [c]_f
    pub len: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct KeccakSpongeCols<T> {
    /// Only used on first row of a round to determine whether the state
    /// prior to absorb should be reset to all 0s.
    /// Constrained to be zero if not first round.
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

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct KeccakMemoryCols<T> {
    pub op_reads: [MemoryReadAuxCols<T, 1>; KECCAK_EXECUTION_READS],
    pub absorb_reads: [MemoryReadAuxCols<T, KECCAK_WORD_SIZE>; KECCAK_ABSORB_READS],
    pub digest_writes: [MemoryWriteAuxCols<T, KECCAK_WORD_SIZE>; KECCAK_DIGEST_WRITES],
    /// The input bytes are batch read in blocks of [KECCAK_WORD_SIZE] bytes. However
    /// if the input length is not a multiple of [KECCAK_WORD_SIZE], we read into
    /// `partial_block` more bytes than we need. On the other hand `block_bytes` expects
    /// only the partial block of bytes and then the correctly padded bytes.
    /// We will select between `partial_block` and `block_bytes` for what to read from memory.
    /// We never read a full padding block, so the first byte is always ok.
    pub partial_block: [T; KECCAK_WORD_SIZE - 1],
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

    pub fn is_first_round(&self) -> T {
        *self.inner.step_flags.first().unwrap()
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
        builder.assert_eq(self.c, other.c);
        builder.assert_eq(self.d, other.d);
        builder.assert_eq(self.e, other.e);
        builder.assert_eq(self.dst, other.dst);
        builder.assert_eq(self.src, other.src);
        builder.assert_eq(self.len, other.len);
    }
}

pub const NUM_KECCAK_VM_COLS: usize = size_of::<KeccakVmCols<u8>>();
pub const NUM_KECCAK_OPCODE_COLS: usize = size_of::<KeccakOpcodeCols<u8>>();
pub const NUM_KECCAK_SPONGE_COLS: usize = size_of::<KeccakSpongeCols<u8>>();
pub const NUM_KECCAK_MEMORY_COLS: usize = size_of::<KeccakMemoryCols<u8>>();
