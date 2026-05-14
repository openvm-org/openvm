use openvm_circuit::system::memory::offline_checker::{MemoryBaseAuxCols, MemoryReadAuxCols};
use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV64_WORD_NUM_LIMBS;

use crate::{KECCAK_WIDTH_MEM_OPS, KECCAK_WIDTH_U16S};

/// Number of u16 cells used to represent the low 32 bits of `buffer_ptr`. The high 4
/// bytes of the 8-byte RV64 register are zero and are hardcoded in the memory bus
/// interaction via `expand_to_rv64_block` rather than materialized as columns. Matches
/// the hintstore `MEM_PTR_NUM_LIMBS` convention.
pub const BUFFER_PTR_NUM_LIMBS: usize = RV64_WORD_NUM_LIMBS / 2;

#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow, StructReflection)]
pub struct KeccakfOpCols<T> {
    /// Program counter
    pub pc: T,
    /// True on the row handling execution for an instruction.
    pub is_valid: T,
    /// The starting timestamp for execution in this row.
    /// A single row will do multiple memory accesses.
    pub timestamp: T,
    /// Pointer to address space 1 `rd` register.
    /// The `rd` register holds the value of `buffer_ptr`.
    pub rd_ptr: T,
    /// `buffer_ptr <- [rd_ptr:8]_1`.
    /// Low 4 bytes of the 8-byte `rd` register packed into 2 u16 cells (matching AS2
    /// u16 cell granularity). Interpreted as the pointer to address space 2 `buffer`.
    /// The upper 4 bytes of the register are zero and are hardcoded in the memory bus
    /// interaction via `expand_to_rv64_block`.
    pub buffer_ptr_limbs: [T; BUFFER_PTR_NUM_LIMBS],
    /// The preimage state, to be permuted in the `keccakf` operation. Stored as
    /// `KECCAK_WIDTH_U16S` u16 cells (one per pair of state bytes) to match the
    /// keccakf periphery bus and AS2 u16-celled memory.
    pub preimage: [T; KECCAK_WIDTH_U16S],
    /// The postimage state after `keccakf` permute of `preimage`. Stored as u16 cells
    /// for the same reason as `preimage`. Note: the byte-pair decompositions are NOT
    /// individually byte-range-checked here because each u16 cell is consumed only in
    /// packed form on the keccakf state bus (canonical u16 produced by the periphery)
    /// and on the memory bus (AS2 is u16-celled).
    ///
    /// Note: there is 2 row per instruction design where these columns can be shared with
    /// `preimage`. However due to the interactions necessary for range checks, currently we
    /// determined it is better to minimum number of rows while using more main columns.
    pub postimage: [T; KECCAK_WIDTH_U16S],
    /// Auxiliary columns for timestamp checking for the read of `[rd_ptr:8]_1`.
    pub rd_aux: MemoryReadAuxCols<T>,
    /// Auxiliary columns for timestamp checking of the writes to `buffer`. The writes are done one
    /// word at a time, and each write requires a separate previous timestamp.
    pub buffer_word_aux: [MemoryBaseAuxCols<T>; KECCAK_WIDTH_MEM_OPS],
}

pub const NUM_KECCAKF_OP_COLS: usize = size_of::<KeccakfOpCols<u8>>();
