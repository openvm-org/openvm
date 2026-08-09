use openvm_circuit::system::memory::offline_checker::{MemoryBaseAuxCols, MemoryReadAuxCols};
use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_poseidon2_air::POSEIDON2_WIDTH;

use crate::{canonicity::CanonicityAuxCols, POSEIDON2_STATE_BYTES};

#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow, StructReflection)]
pub struct Poseidon2PermuteOpCols<T> {
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
    /// `buffer_ptr <- [rd_ptr:4]_1`.
    /// Limbs of the pointer to address space 2 `buffer`.
    pub buffer_ptr_limbs: [T; RV32_REGISTER_NUM_LIMBS],
    /// The preimage state, to be permuted in the `PERMUTE` operation.
    pub preimage: [T; POSEIDON2_STATE_BYTES],
    /// The postimage state after `PERMUTE` of `preimage`.
    pub postimage: [T; POSEIDON2_STATE_BYTES],
    /// Auxiliary columns for timestamp checking for the read of `[rd_ptr:4]_1`.
    pub rd_aux: MemoryReadAuxCols<T>,
    /// Auxiliary columns for timestamp checking of the writes to `buffer`. The writes are done one
    /// word at a time, and each write requires a separate previous timestamp.
    pub buffer_word_aux: [MemoryBaseAuxCols<T>; POSEIDON2_WIDTH],
    /// Auxiliary columns to constrain that each `postimage` word is a *canonical* byte
    /// decomposition of the field element sent on the poseidon2 bus. Without this the bytes
    /// written back to memory would be under-constrained; see the AIR for details.
    pub postimage_canonicity_aux: [CanonicityAuxCols<T>; POSEIDON2_WIDTH],
}

pub const NUM_POSEIDON2_PERMUTE_OP_COLS: usize = size_of::<Poseidon2PermuteOpCols<u8>>();
