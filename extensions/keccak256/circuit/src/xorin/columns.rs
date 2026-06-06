use openvm_circuit::{
    arch::BLOCK_FE_WIDTH,
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_circuit::adapters::RV64_PTR_U16_LIMBS;

use crate::{KECCAK_RATE_BYTES, KECCAK_RATE_MEM_OPS};

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
    /// Low 32 bits of the `rs0` register as u16 cells.
    pub buffer_ptr_limbs: [T; RV64_PTR_U16_LIMBS],
    /// Low 32 bits of the `rs1` register as u16 cells.
    pub input_ptr_limbs: [T; RV64_PTR_U16_LIMBS],
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
    /// Carry for converting the base `buffer`/`input` *byte* pointers to AS-native u16 *cell*
    /// pointer limbs.
    pub buffer_cell_carry: T,
    pub input_cell_carry: T,
    /// Per-block carry for adding the cell offset `i * (MEMORY_BLOCK_BYTES / U16_CELL_SIZE)` to
    /// each base cell pointer (block `i`'s carry into the high cell limb). One set per heap
    /// access group (buffer read, input read, buffer write).
    pub buffer_read_add_carry: [T; KECCAK_RATE_MEM_OPS],
    pub input_read_add_carry: [T; KECCAK_RATE_MEM_OPS],
    pub buffer_write_add_carry: [T; KECCAK_RATE_MEM_OPS],
}

pub const NUM_XORIN_VM_COLS: usize = size_of::<XorinVmCols<u8>>();
