use openvm_circuit::system::memory::offline_checker::{MemoryBaseAuxCols, MemoryReadAuxCols};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;

use crate::{KECCAK_WIDTH_BYTES, KECCAK_WIDTH_WORDS};

/// Each instruction handling will use _two_ rows.
#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow)]
pub struct KeccakfOpCols<T> {
    /// Program counter
    pub pc: T,
    /// True on the row handling execution for an instruction.
    pub is_valid: T,
    /// True only on the row immediately after an `is_valid = true` row.
    pub is_after_valid: T,
    /// The starting timestamp for execution in this row.
    /// A single row will do multiple memory accesses.
    pub timestamp: T,
    /// Pointer to address space 1 `rd` register.
    /// The `rd` register holds the value of `buffer_ptr`.
    pub rd_ptr: T,
    /// `buffer_ptr <- [rd_ptr:4]_1`.
    /// Limbs of the pointer to address space 2 `buffer`.
    pub buffer_ptr_limbs: [T; RV32_REGISTER_NUM_LIMBS],
    /// The buffer itself, to be permuted in the `keccakf` operation.
    /// The AIR design is such that opcode execution will use _two_ rows:
    /// 1. The first row with `is_enabled = true` will contain the read previous `buffer` values.
    /// 2. The second row with `is_enabled = false` will contain the values after the keccakf
    ///    permutation.
    pub buffer: [T; KECCAK_WIDTH_BYTES],
    /// Auxiliary columns for timestamp checking for the read of `[rd_ptr:4]_1`.
    pub rd_aux: MemoryReadAuxCols<T>,
    /// Auxiliary columns for timestamp checking of the writes to `buffer`. The writes are done one
    /// word at a time, and each write requires a separate previous timestamp.
    pub buffer_word_aux: [MemoryBaseAuxCols<T>; KECCAK_WIDTH_WORDS],
}

pub const NUM_KECCAKF_OP_COLS: usize = size_of::<KeccakfOpCols<u8>>();
