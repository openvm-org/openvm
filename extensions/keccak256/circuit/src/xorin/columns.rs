use openvm_circuit::{
    arch::BLOCK_FE_WIDTH,
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::RV64_WORD_NUM_LIMBS;

use crate::{KECCAK_RATE_BYTES, KECCAK_RATE_MEM_OPS};

/// Number of u16 cells used to represent the low 32 bits of each pointer
/// (`buffer_ptr` / `input_ptr`). The high 4 bytes of the 8-byte RV64 register are zero
/// and hardcoded in the memory bus interaction via `expand_to_rv64_block` rather than
/// materialized as columns. Matches the `BUFFER_PTR_NUM_LIMBS` convention used by
/// `keccakf_op` and hintstore's `MEM_PTR_NUM_LIMBS`.
pub const XORIN_PTR_NUM_LIMBS: usize = RV64_WORD_NUM_LIMBS / 2;

#[repr(C)]
#[derive(Debug, AlignedBorrow, StructReflection)]
pub struct XorinVmCols<T> {
    pub sponge: XorinSpongeCols<T>,
    pub instruction: XorinInstructionCols<T>,
    pub mem_oc: XorinMemoryCols<T>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, StructReflection, derive_new::new)]
#[allow(clippy::too_many_arguments)]
pub struct XorinInstructionCols<T> {
    pub pc: T,
    pub is_enabled: T,
    pub buffer_reg_ptr: T,
    pub input_reg_ptr: T,
    pub len_reg_ptr: T,
    pub buffer_ptr: T,
    /// Low 4 bytes of the 8-byte `rs0` register packed into 2 u16 cells (matching AS2
    /// u16 cell granularity). The upper 4 bytes of the register are zero and are
    /// hardcoded in the memory bus interaction via `expand_to_rv64_block`.
    pub buffer_ptr_limbs: [T; XORIN_PTR_NUM_LIMBS],
    pub input_ptr: T,
    /// See `buffer_ptr_limbs`.
    pub input_ptr_limbs: [T; XORIN_PTR_NUM_LIMBS],
    pub len: T,
    pub len_limb: T,
    pub start_timestamp: T,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow, StructReflection)]
pub struct XorinSpongeCols<T> {
    // is_padding_bytes is a boolean where is_padding_bytes[i] = 1 if the i-th 8-byte memory
    // block is padding and 0 otherwise.
    pub is_padding_bytes: [T; KECCAK_RATE_MEM_OPS],
    pub preimage_buffer_bytes: [T; KECCAK_RATE_BYTES],
    pub input_bytes: [T; KECCAK_RATE_BYTES],
    pub postimage_buffer_bytes: [T; KECCAK_RATE_BYTES],
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow, StructReflection)]
pub struct XorinMemoryCols<T> {
    pub register_aux_cols: [MemoryReadAuxCols<T>; 3],
    pub input_bytes_read_aux_cols: [MemoryReadAuxCols<T>; KECCAK_RATE_MEM_OPS],
    pub buffer_bytes_read_aux_cols: [MemoryReadAuxCols<T>; KECCAK_RATE_MEM_OPS],
    pub buffer_bytes_write_aux_cols: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; KECCAK_RATE_MEM_OPS],
}

pub const NUM_XORIN_VM_COLS: usize = size_of::<XorinVmCols<u8>>();
