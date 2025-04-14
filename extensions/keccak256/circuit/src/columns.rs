use core::mem::size_of;

use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_air::AirBuilder;
use p3_keccak_air::KeccakCols as KeccakPermCols;

use super::{
    KECCAK_ABSORB_READS, KECCAK_DIGEST_WRITES, KECCAK_RATE_BYTES, KECCAK_RATE_U16S,
    KECCAK_REGISTER_READS, KECCAK_WORD_SIZE,
};

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakVmCols<T> {
    /// Columns for keccak-f permutation
    pub inner: KeccakPermCols<T>,
    /// Columns for sponge and padding
    pub sponge: KeccakSpongeCols<T>,
    /// Columns for instruction interface and register access
    pub instruction: KeccakInstructionCols<T>,
    /// Auxiliary columns for offline memory checking
    pub mem_oc: KeccakMemoryCols<T>,
}

/// Columns for KECCAK256_RV32 instruction parsing.
/// Includes columns for instruction execution and register reads.
#[allow(clippy::too_many_arguments)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct KeccakInstructionCols<T> {
    /// Program counter
    pub pc: T,
    /// True for all rows that are part of opcode execution.
    /// False on dummy rows only used to pad the height.
    pub is_enabled: T,
    /// Is enabled and first round of block. Used to lower constraint degree.
    /// is_enabled * inner.step_flags\[0\]
    pub is_enabled_first_round: T,
    /// The starting timestamp to use for memory access in this row.
    /// A single row will do multiple memory accesses.
    pub start_timestamp: T,
    /// Pointer to address space 1 `dst` register
    pub dst_ptr: T,
    /// Pointer to address space 1 `src` register
    pub src_ptr: T,
    /// Pointer to address space 1 `len` register
    pub len_ptr: T,
    // Register values
    /// dst <- \[dst_ptr:4\]_1
    pub dst: [T; RV32_REGISTER_NUM_LIMBS],
    /// src <- \[src_ptr:4\]_1
    /// We store src_limbs\[i\] = \[src_ptr + i + 1\]_1 and src = u32(\[src_ptr:4\]_1) from which
    /// \[src_ptr\]_1 can be recovered by linear combination.
    /// We do this because `src` needs to be incremented between keccak-f permutations.
    pub src_limbs: [T; RV32_REGISTER_NUM_LIMBS - 1],
    pub src: T,
    /// len <- \[len_ptr:4\]_1
    /// We store len_limbs\[i\] = \[len_ptr + i + 1\]_1 and remaining_len = u32(\[len_ptr:4\]_1)
    /// from which \[len_ptr\]_1 can be recovered by linear combination.
    /// We do this because `remaining_len` needs to be decremented between keccak-f permutations.
    pub len_limbs: [T; RV32_REGISTER_NUM_LIMBS - 1],
    /// The remaining length of the unpadded input, in bytes.
    /// If `is_new_start` is true and `is_enabled` is true, this must be equal to `u32(len)`.
    pub remaining_len: T,
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
    pub register_aux: [MemoryReadAuxCols<T>; KECCAK_REGISTER_READS],
    pub absorb_reads: [MemoryReadAuxCols<T>; KECCAK_ABSORB_READS],
    pub digest_writes: [MemoryWriteAuxCols<T, KECCAK_WORD_SIZE>; KECCAK_DIGEST_WRITES],
    /// The input bytes are batch read in blocks of private constant KECCAK_WORD_SIZE bytes.
    /// However if the input length is not a multiple of KECCAK_WORD_SIZE, we read into
    /// `partial_block` more bytes than we need. On the other hand `block_bytes` expects
    /// only the partial block of bytes and then the correctly padded bytes.
    /// We will select between `partial_block` and `block_bytes` for what to read from memory.
    /// We never read a full padding block, so the first byte is always ok.
    pub partial_block: [T; KECCAK_WORD_SIZE - 1],
}

impl<T: Copy> KeccakVmCols<T> {
    pub const fn remaining_len(&self) -> T {
        self.instruction.remaining_len
    }

    pub const fn is_new_start(&self) -> T {
        self.sponge.is_new_start
    }

    pub fn postimage(&self, y: usize, x: usize, limb: usize) -> T {
        self.inner.a_prime_prime_prime(y, x, limb)
    }

    pub fn is_first_round(&self) -> T {
        *self.inner.step_flags.first().unwrap()
    }

    pub fn is_last_round(&self) -> T {
        *self.inner.step_flags.last().unwrap()
    }
}

impl<T: Copy> KeccakInstructionCols<T> {
    pub fn assert_eq<AB: AirBuilder>(&self, builder: &mut AB, other: Self)
    where
        T: Into<AB::Expr>,
    {
        builder.assert_eq(self.pc, other.pc);
        builder.assert_eq(self.is_enabled, other.is_enabled);
        builder.assert_eq(self.start_timestamp, other.start_timestamp);
        builder.assert_eq(self.dst_ptr, other.dst_ptr);
        builder.assert_eq(self.src_ptr, other.src_ptr);
        builder.assert_eq(self.len_ptr, other.len_ptr);
        assert_array_eq(builder, self.dst, other.dst);
        assert_array_eq(builder, self.src_limbs, other.src_limbs);
        builder.assert_eq(self.src, other.src);
        assert_array_eq(builder, self.len_limbs, other.len_limbs);
        builder.assert_eq(self.remaining_len, other.remaining_len);
    }
}

pub const NUM_KECCAK_VM_COLS: usize = size_of::<KeccakVmCols<u8>>();
pub const NUM_KECCAK_INSTRUCTION_COLS: usize = size_of::<KeccakInstructionCols<u8>>();
pub const NUM_KECCAK_SPONGE_COLS: usize = size_of::<KeccakSpongeCols<u8>>();
pub const NUM_KECCAK_MEMORY_COLS: usize = size_of::<KeccakMemoryCols<u8>>();
