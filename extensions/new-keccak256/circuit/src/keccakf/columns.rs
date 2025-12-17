use openvm_circuit::system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols};
use openvm_circuit_primitives_derive::AlignedBorrow;
use p3_keccak_air::KeccakCols as KeccakPermCols;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakfVmCols<T> {
    pub inner: KeccakPermCols<T>,
    pub preimage_state_hi: [T; 100],
    pub postimage_state_hi: [T; 100],
    pub instruction: KeccakfInstructionCols<T>,
    pub mem_oc: KeccakfMemoryCols<T>,
    pub timestamp: T,
    pub is_enabled_is_first_round: T,
    pub is_enabled_is_final_round: T,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct KeccakfInstructionCols<T> {
    pub pc: T,
    pub is_enabled: T, 
    pub buffer_ptr: T,
    pub buffer: T,
    pub buffer_limbs: [T; 4],
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct KeccakfMemoryCols<T> {
    pub register_aux_cols: [MemoryReadAuxCols<T>; 1],
    pub buffer_bytes_read_aux_cols: [MemoryReadAuxCols<T>; 200/4],
    pub buffer_bytes_write_aux_cols: [MemoryWriteAuxCols<T, 4>; 200/4],
}

pub const NUM_KECCAKF_VM_COLS: usize = size_of::<KeccakfVmCols<u8>>();
pub const NUM_KECCAK_PERM_COLS: usize = size_of::<KeccakPermCols<u8>>();
pub const NUM_ROUNDS: usize = 24;