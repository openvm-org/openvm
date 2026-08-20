use openvm_circuit::system::memory::offline_checker::{MemoryBaseAuxCols, MemoryReadAuxCols};
use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_circuit::adapters::PTR_U16_LIMBS;

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
    /// Low 32 bits of the buffer register as u16 cells.
    pub buffer_ptr_limbs: [T; PTR_U16_LIMBS],
    /// Low 32 bits of the input register as u16 cells.
    pub input_ptr_limbs: [T; PTR_U16_LIMBS],
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
    // Only store write timestamp auxiliaries; previous data comes from preimage_buffer_bytes.
    pub buffer_bytes_write_base_aux: [MemoryBaseAuxCols<T>; KECCAK_RATE_MEM_OPS],
    /// Carry for converting the base `buffer`/`input` *byte* pointers to AS-native u16 *cell*
    /// pointer limbs.
    pub buffer_cell_carry: T,
    pub input_cell_carry: T,
}

pub const NUM_XORIN_VM_COLS: usize = size_of::<XorinVmCols<u8>>();
