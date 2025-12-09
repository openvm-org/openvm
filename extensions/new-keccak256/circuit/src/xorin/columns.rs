use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives_derive::AlignedBorrow;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct XorinVmCols<T> {
    pub sponge: XorinSpongeCols<T>,
    pub instruction: XorinInstructionCols<T>,
    pub mem_oc: XorinMemoryCols<T>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct XorinInstructionCols<T> {
    pub pc: T, 
    pub is_enabled: T,
    pub buffer_ptr: T,
    pub input_ptr: T,
    pub len_ptr: T,
    pub buffer_limbs: [T; 2],
    pub input_limbs: [T; 2],
    pub len_limbs: [T; 2],
    pub timestamp: T
}

#[repr(C)]
#[derive(Copy, Clone, Debug, AlignedBorrow)]
pub struct XorinSpongeCols<T> {
    pub is_padding_bytes: [T; 136],
    pub preimage_buffer_bytes: [T; 136],
    pub input_bytes: [T; 136],
    pub postimage_buffer_bytes: [T; 136],
}


#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct XorinMemoryCols<T> {
    pub register_aux_cols: [MemoryReadAuxCols<T>; 3],
    pub buffer_bytes_read_aux_cols: [MemoryReadAuxCols<T>; 34],
    pub buffer_bytes_write_aux_cols: [MemoryWriteAuxCols<T, 4>; 34],
}