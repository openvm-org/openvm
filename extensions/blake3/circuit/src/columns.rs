use core::mem::size_of;

use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_air::AirBuilder;
use p3_blake3_air::Blake3Cols as Blake3CompressCols;

use super::{BLAKE3_DIGEST_WRITES, BLAKE3_INPUT_READS, BLAKE3_REGISTER_READS, BLAKE3_WORD_SIZE};

/// Main columns struct for BLAKE3 VM integration.
/// Wraps the p3-blake3-air compression columns and adds VM-specific columns.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Blake3VmCols<T> {
    /// Columns for blake3 compression function from p3-blake3-air.
    /// One compression per row.
    pub inner: Blake3CompressCols<T>,
    /// Columns for instruction interface and register access.
    pub instruction: Blake3InstructionCols<T>,
    /// Auxiliary columns for offline memory checking.
    pub mem_oc: Blake3MemoryCols<T>,
}

/// Columns for BLAKE3 instruction parsing and multi-block state tracking.
/// Includes columns for instruction execution and register reads.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct Blake3InstructionCols<T> {
    /// Program counter.
    pub pc: T,
    /// True for all rows that are part of opcode execution.
    /// False on dummy rows only used to pad the height.
    pub is_enabled: T,
    /// Is enabled and first block of hash. Used to lower constraint degree.
    /// is_enabled * is_new_start
    pub is_enabled_first_block: T,
    /// The starting timestamp to use for memory access in this row.
    /// A single row will do multiple memory accesses.
    pub start_timestamp: T,
    /// Pointer to address space 1 `dst` register.
    pub dst_ptr: T,
    /// Pointer to address space 1 `src` register.
    pub src_ptr: T,
    /// Pointer to address space 1 `len` register.
    pub len_ptr: T,
    // Register values
    /// dst <- [dst_ptr:4]_1 (output destination pointer).
    pub dst: [T; RV32_REGISTER_NUM_LIMBS],
    /// src <- [src_ptr:4]_1
    /// We store src_limbs[i] = [src_ptr + i + 1]_1 and src = u32([src_ptr:4]_1) from which
    /// [src_ptr]_1 can be recovered by linear combination.
    /// We do this because `src` needs to be incremented between blocks.
    pub src_limbs: [T; RV32_REGISTER_NUM_LIMBS - 1],
    pub src: T,
    /// len <- [len_ptr:4]_1
    /// We store len_limbs[i] = [len_ptr + i + 1]_1 and remaining_len = u32([len_ptr:4]_1)
    /// from which [len_ptr]_1 can be recovered by linear combination.
    /// We do this because `remaining_len` needs to be decremented between blocks.
    pub len_limbs: [T; RV32_REGISTER_NUM_LIMBS - 1],
    /// The remaining length of the unpadded input, in bytes.
    /// If `is_new_start` is true and `is_enabled` is true, this must equal the original length.
    pub remaining_len: T,
    /// Whether this is the first block of a new hash operation.
    /// Used to determine if chaining value should be IV.
    pub is_new_start: T,
    /// Whether this is the last block of a hash operation.
    /// Used to determine if we should write output to memory.
    pub is_last_block: T,
}

/// Auxiliary columns for offline memory checking.
#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct Blake3MemoryCols<T> {
    /// Auxiliary columns for reading dst, src, len registers.
    pub register_aux: [MemoryReadAuxCols<T>; BLAKE3_REGISTER_READS],
    /// Auxiliary columns for reading input data (64 bytes = 16 words per block).
    pub input_reads: [MemoryReadAuxCols<T>; BLAKE3_INPUT_READS],
    /// Auxiliary columns for writing digest output (32 bytes = 8 words).
    pub digest_writes: [MemoryWriteAuxCols<T, BLAKE3_WORD_SIZE>; BLAKE3_DIGEST_WRITES],
    /// The input bytes are batch read in blocks of BLAKE3_WORD_SIZE bytes.
    /// However if the input length is not a multiple of BLAKE3_WORD_SIZE, we read
    /// more bytes than we need. This stores the extra bytes for the last partial read.
    /// We never read a full padding block, so the first byte is always valid.
    pub partial_block: [T; BLAKE3_WORD_SIZE - 1],
}

impl<T: Copy> Blake3VmCols<T> {
    /// Get the remaining length of unprocessed input.
    pub const fn remaining_len(&self) -> T {
        self.instruction.remaining_len
    }

    /// Check if this is the first block of a new hash.
    pub const fn is_new_start(&self) -> T {
        self.instruction.is_new_start
    }

    /// Check if this is the last block of the hash.
    pub const fn is_last_block(&self) -> T {
        self.instruction.is_last_block
    }
}

impl<T: Copy> Blake3InstructionCols<T> {
    /// Assert equality between two instruction column sets.
    /// Used for constraining that multi-block hashes maintain consistent instruction state.
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

pub const NUM_BLAKE3_VM_COLS: usize = size_of::<Blake3VmCols<u8>>();
pub const NUM_BLAKE3_INSTRUCTION_COLS: usize = size_of::<Blake3InstructionCols<u8>>();
pub const NUM_BLAKE3_MEMORY_COLS: usize = size_of::<Blake3MemoryCols<u8>>();
